"""
aux_verificar_multioutput.py
============================
Script de verificación autónomo (no toca el repo ni los datos).

Decide si es viable migrar step005 al modelo composite multi-output que propone
el paper 2406.02293: UN solo booster con hojas vectoriales que predice los 7
cuantiles a la vez, en vez de 7 boosters independientes.

La pregunta que responde: ¿funciona `multi_strategy="multi_output_tree"` de
XGBoost combinado con un objetivo CUSTOM? Es una combinación poco documentada.

Datos sintéticos con cuantiles verdaderos conocidos, y con el MISMO tamaño de
muestra del problema real (~500 train / ~121 val): probar con 10,000 puntos
daría una respuesta engañosamente optimista.

Uso:  python aux_verificar_multioutput.py
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuración del experimento — refleja el régimen real de step005
# ---------------------------------------------------------------------------
TAUS       = [0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99]
N_TRAIN    = 500      # ~ filas de TRAIN por (fold, h)
N_VAL      = 121      # ~ filas de VAL por (fold, h)
N_FEAT     = 20
N_ROUNDS   = 150      # ~ n_estimators típico que elige Optuna
S_FACTOR   = 0.5      # ~ s_factor típico que elige Optuna
SEED       = 42

_OK, _FAIL, _WARN = "[ OK ]", "[FALLA]", "[AVISO]"
resultados: dict[str, bool] = {}


def _titulo(txt: str) -> None:
    print(f"\n{'=' * 74}\n{txt}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# Datos: Y | X ~ N(sin(7·x0), 0.2²)  → cuantil verdadero = sin(7x0) + 0.2·z_tau
# ---------------------------------------------------------------------------
def _z(p: float) -> float:
    """Cuantil de la normal estándar (aprox. Acklam, sin scipy)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def generar(n: int, escala: float, rng) -> tuple:
    """Devuelve X, y, y la matriz (n, 7) de cuantiles verdaderos."""
    X  = rng.random((n, N_FEAT))
    mu = np.sin(7 * X[:, 0])
    sd = 0.2
    y  = rng.normal(mu, sd)
    q_true = np.column_stack([mu + sd * _z(t) for t in TAUS])
    return X * escala, y * escala, q_true * escala


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


# ---------------------------------------------------------------------------
# Objetivos: el de step005 (una columna) y su versión vectorizada (n, 7)
# ---------------------------------------------------------------------------
def obj_single(tau: float, s: float, std_y: float):
    _scale = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)

    def f(y_pred, dtrain):
        y_pred = np.clip(y_pred, -1e15, 1e15)
        u    = dtrain.get_label() - y_pred
        s2u2 = s ** 2 + u ** 2
        grad = -((tau - 0.5 + np.arctan(u / s) / np.pi)
                 + u * s / (np.pi * s2u2)) * _scale
        hess = ((s ** 2 + std_y ** 2) / s2u2) ** 2
        return grad, hess
    return f


def obj_multi(taus, s: float, std_y: float, traza: dict | None = None):
    taus_arr = np.asarray(taus, dtype=float).reshape(1, -1)
    _scale   = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)

    def f(y_pred, dtrain):
        if traza is not None and "predt_shape" not in traza:
            traza["predt_shape"] = np.shape(y_pred)
            traza["label_shape"] = np.shape(dtrain.get_label())
        y_pred = np.clip(np.asarray(y_pred, dtype=float), -1e15, 1e15)
        y      = dtrain.get_label().reshape(y_pred.shape)
        u      = y - y_pred
        s2u2   = s ** 2 + u ** 2
        grad   = -((taus_arr - 0.5 + np.arctan(u / s) / np.pi)
                   + u * s / (np.pi * s2u2)) * _scale
        hess   = ((s ** 2 + std_y ** 2) / s2u2) ** 2
        return grad, hess
    return f


PARAMS_BASE = {
    "tree_method"     : "hist",
    "max_bin"         : 64,
    "max_depth"       : 3,
    "learning_rate"   : 0.05,
    "min_child_weight": 5,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda"      : 2.0,
    "nthread"         : 1,
    "seed"            : SEED,
}


def entrenar_multi(X, y, s, std_y, traza=None):
    Y = np.tile(np.asarray(y, dtype=float).reshape(-1, 1), (1, len(TAUS)))
    d = xgb.DMatrix(X, label=Y)
    p = dict(PARAMS_BASE, multi_strategy="multi_output_tree")
    bst = xgb.train(p, d, num_boost_round=N_ROUNDS,
                    obj=obj_multi(TAUS, s, std_y, traza), verbose_eval=False)
    return bst


def entrenar_separados(X, y, s, std_y):
    d = xgb.DMatrix(X, label=y)
    return [xgb.train(PARAMS_BASE, d, num_boost_round=N_ROUNDS,
                      obj=obj_single(t, s, std_y), verbose_eval=False)
            for t in TAUS]


def cruces(Q: np.ndarray) -> tuple[int, int]:
    """(filas con al menos un cruce, violaciones totales)."""
    dif = np.diff(Q, axis=1)
    return int((dif < 0).any(axis=1).sum()), int((dif < 0).sum())


# ===========================================================================
def main() -> None:
    print(f"xgboost {xgb.__version__}   |   numpy {np.__version__}")
    print(f"config: n_train={N_TRAIN}  n_val={N_VAL}  n_feat={N_FEAT}  "
          f"rondas={N_ROUNDS}  taus={len(TAUS)}")

    rng = np.random.default_rng(SEED)

    # ── T1 · ¿existe multi_strategy? ──────────────────────────────────────
    _titulo("T1 · Disponibilidad de multi_strategy")
    ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if ver < (2, 0):
        print(f"{_FAIL} multi_strategy requiere XGBoost >= 2.0 (tienes {xgb.__version__}).")
        print("       Sin esto el plan multi-output no es viable con esta instalación.")
        resultados["T1"] = False
        return
    print(f"{_OK} XGBoost {xgb.__version__} >= 2.0")
    resultados["T1"] = True

    # ── T2 · objetivo custom + multi_output_tree → shape (n, 7) ───────────
    _titulo("T2 · Objetivo custom sobre multi_output_tree")
    X, y, q_true = generar(N_TRAIN, 1.0, rng)
    std_y = max(float(np.std(y)), 1.0)
    s = S_FACTOR * std_y
    traza: dict = {}
    try:
        bst = entrenar_multi(X, y, s, std_y, traza)
        P = bst.predict(xgb.DMatrix(X))
        print(f"   shape que recibe el objetivo : predt={traza.get('predt_shape')}  "
              f"label={traza.get('label_shape')}")
        print(f"   shape de predict()           : {P.shape}")
        if P.shape == (N_TRAIN, len(TAUS)):
            print(f"{_OK} devuelve (n, {len(TAUS)}) — la API funciona")
            resultados["T2"] = True
        else:
            print(f"{_FAIL} se esperaba ({N_TRAIN}, {len(TAUS)})")
            resultados["T2"] = False
            return
    except Exception:
        print(f"{_FAIL} excepción al entrenar:")
        traceback.print_exc()
        resultados["T2"] = False
        return

    # ── T3 · reshape del label: ¿row-major correcto? ──────────────────────
    _titulo("T3 · Orden del label al hacer reshape")
    Y2 = np.tile(y.reshape(-1, 1), (1, len(TAUS)))
    Y2[:, 0] = -999.0                      # marca la primera columna
    d2 = xgb.DMatrix(X, label=Y2)
    lab = d2.get_label().reshape(-1, len(TAUS))
    if np.allclose(lab[:, 0], -999.0):
        print(f"{_OK} reshape(n, {len(TAUS)}) recupera las columnas correctamente")
        resultados["T3"] = True
    else:
        print(f"{_FAIL} el reshape NO es row-major — el objetivo mezclaría cuantiles")
        print(f"        primera columna recuperada: {lab[:5, 0]}")
        resultados["T3"] = False

    # ── T4 · ¿aprende cuantiles de verdad? ────────────────────────────────
    _titulo("T4 · Calibración contra los cuantiles verdaderos (VAL)")
    Xv, yv, qv_true = generar(N_VAL, 1.0, rng)
    Pv_multi = bst.predict(xgb.DMatrix(Xv))
    seps     = entrenar_separados(X, y, s, std_y)
    Pv_sep   = np.column_stack([b.predict(xgb.DMatrix(Xv)) for b in seps])

    print(f"   {'tau':>5} {'cobertura real':>15} {'multi':>9} {'separados':>11}")
    print("   " + "-" * 44)
    for i, t in enumerate(TAUS):
        cob_m = float((yv <= Pv_multi[:, i]).mean())
        cob_s = float((yv <= Pv_sep[:, i]).mean())
        print(f"   {t:>5.2f} {t:>15.2f} {cob_m:>9.3f} {cob_s:>11.3f}")
    ecm = float(np.mean([abs((yv <= Pv_multi[:, i]).mean() - t)
                         for i, t in enumerate(TAUS)]))
    ecs = float(np.mean([abs((yv <= Pv_sep[:, i]).mean() - t)
                         for i, t in enumerate(TAUS)]))
    print(f"\n   ECE  multi={ecm:.4f}   separados={ecs:.4f}")
    resultados["T4"] = ecm < 0.25          # umbral laxo: solo verifica que aprende algo
    print(f"{_OK if resultados['T4'] else _WARN} "
          f"{'aprende cuantiles' if resultados['T4'] else 'calibración pobre — revisar'}")

    # ── T5 · cruces: la promesa central del paper ─────────────────────────
    _titulo("T5 · Cruces de cuantiles (la promesa del paper)")
    fm, vm = cruces(Pv_multi)
    fs, vs = cruces(Pv_sep)
    print(f"   multi-output : {fm:>4}/{N_VAL} filas con cruce  ({vm} violaciones)")
    print(f"   7 separados  : {fs:>4}/{N_VAL} filas con cruce  ({vs} violaciones)")
    if fs > 0:
        print(f"   reducción    : {100 * (1 - fm / max(fs, 1)):.0f}%")
    resultados["T5"] = fm <= fs
    print(f"{_OK if resultados['T5'] else _WARN} "
          f"{'multi-output cruza menos o igual' if resultados['T5'] else 'multi-output cruza MÁS — inesperado'}")

    # ── T6 · escala real del target (~1e9) ────────────────────────────────
    _titulo("T6 · Escala del target real (~1e9), donde _scale ~ 1e10")
    try:
        Xb, yb, _ = generar(N_TRAIN, 5e8, rng)
        sb = S_FACTOR * max(float(np.std(yb)), 1.0)
        bb = entrenar_multi(Xb, yb, sb, max(float(np.std(yb)), 1.0))
        Pb = bb.predict(xgb.DMatrix(Xb))
        fin = bool(np.isfinite(Pb).all())
        print(f"   predicciones finitas: {fin}   rango: "
              f"[{np.nanmin(Pb):.3e}, {np.nanmax(Pb):.3e}]")
        print(f"   escala del target   : [{yb.min():.3e}, {yb.max():.3e}]")
        resultados["T6"] = fin
        print(f"{_OK if fin else _FAIL} "
              f"{'sin overflow' if fin else 'overflow / NaN con la escala real'}")
    except Exception:
        print(f"{_FAIL} excepción:")
        traceback.print_exc()
        resultados["T6"] = False

    # ── T7 · tiempo ───────────────────────────────────────────────────────
    _titulo("T7 · Tiempo de entrenamiento")
    t0 = time.time(); entrenar_multi(X, y, s, std_y);      t_m = time.time() - t0
    t0 = time.time(); entrenar_separados(X, y, s, std_y);  t_s = time.time() - t0
    print(f"   multi-output : {t_m:.2f} s")
    print(f"   7 separados  : {t_s:.2f} s")
    print(f"   speedup      : {t_s / max(t_m, 1e-9):.1f}x")
    resultados["T7"] = True

    # ── T8 · qué pasa con los diagnósticos ────────────────────────────────
    _titulo("T8 · Diagnósticos: gain y SHAP sobre el árbol multi-output")
    try:
        sc = bst.get_score(importance_type="total_gain")
        print(f"   get_score() devuelve {len(sc)} entradas "
              f"(un único set, NO uno por cuantil)")
        print("   → el gain por τ deja de existir: los splits son compartidos")
        resultados["T8_gain"] = True
    except Exception as e:
        print(f"{_WARN} get_score() falló: {e}")
        resultados["T8_gain"] = False

    try:
        import shap
        print(f"   shap {shap.__version__}")
        ex = shap.TreeExplainer(bst)
        sv = ex.shap_values(Xv[:20], check_additivity=False)
        forma = np.shape(sv) if not isinstance(sv, list) else f"lista de {len(sv)}"
        print(f"   shap_values shape: {forma}")
        print(f"{_OK} SHAP funciona sobre el booster multi-output")
        resultados["T8_shap"] = True
    except ImportError:
        print(f"{_WARN} shap no instalado — no se puede verificar")
        resultados["T8_shap"] = None
    except Exception as e:
        print(f"{_WARN} SHAP falló sobre multi-output: {type(e).__name__}: {e}")
        print("        (habría que recalcular SHAP por cuantil de otra forma)")
        resultados["T8_shap"] = False

    # ── Veredicto ─────────────────────────────────────────────────────────
    _titulo("VEREDICTO")
    criticos = ["T1", "T2", "T3", "T6"]
    ok_crit  = all(resultados.get(k) for k in criticos)
    for k in ("T1", "T2", "T3", "T4", "T5", "T6", "T8_gain", "T8_shap"):
        if k in resultados:
            v = resultados[k]
            marca = "?" if v is None else ("si" if v else "NO")
            print(f"   {k:<9} {marca}")
    print()
    if ok_crit:
        print("  VIABLE: la API soporta objetivo custom sobre multi_output_tree")
        print("  a la escala real del problema. Se puede proceder con la migración.")
        if resultados.get("T8_shap") is False:
            print("  Salvedad: SHAP requiere trabajo adicional.")
    else:
        fallos = [k for k in criticos if not resultados.get(k)]
        print(f"  NO VIABLE con esta instalación. Fallaron: {', '.join(fallos)}")
        print("  Alternativa: la implementación de referencia del paper en")
        print("  github.com/LaurensSluyterman/XGBoost_quantile_regression")


if __name__ == "__main__":
    main()
