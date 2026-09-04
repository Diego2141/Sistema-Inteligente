"""
aux_verificar_multioutput.py
============================
Script de verificación autónomo (no toca el repo ni los datos).

Decide QUÉ ruta hacia el modelo composite multi-output del paper 2406.02293
es viable con la instalación de XGBoost disponible. Hay tres candidatas:

  RUTA A · objetivo custom (arctan) + multi_strategy="multi_output_tree"
           Lo ideal: conserva la pérdida arctan Y gana los splits compartidos.
           Requiere que xgb.Booster.boost() acepte gradientes 2-D. En XGBoost
           2.0.3 NO los acepta (c_array solo maneja 1-D) — llegó después.

  RUTA B · reg:quantileerror con quantile_alpha=[lista] + multi_output_tree
           Nativo desde XGBoost 2.0. Splits compartidos sin objetivo custom.
           Se pierde el suavizado arctan (vuelve el gradiente binario con
           hessiano surrogate), pero se gana la estructura composite — que es
           justo lo que ataca el ruido de las colas.

  RUTA C · reg:quantileerror con quantile_alpha=[lista], un árbol por salida
           Multi-output en la API pero SIN splits compartidos. Equivale a lo
           que ya hace step005; sirve de control.

Datos sintéticos con cuantiles verdaderos conocidos y con el MISMO tamaño de
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
TAUS     = [0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99]
N_TRAIN  = 500      # ~ filas de TRAIN por (fold, h) en step005
N_VAL    = 121      # ~ filas de VAL por (fold, h)
N_FEAT   = 20
N_ROUNDS = 150
S_FACTOR = 0.5
SEED     = 42

_OK, _FAIL, _WARN = "[ OK ]", "[FALLA]", "[AVISO]"
R: dict = {}


def _titulo(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# Datos: Y | X ~ N(sin(7·x0), 0.2²) → cuantil verdadero = sin(7x0) + 0.2·z_tau
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


def generar(n: int, escala: float, rng):
    X  = rng.random((n, N_FEAT))
    mu = np.sin(7 * X[:, 0])
    sd = 0.2
    y  = rng.normal(mu, sd)
    q  = np.column_stack([mu + sd * _z(t) for t in TAUS])
    return X * escala, y * escala, q * escala


def cruces(Q: np.ndarray):
    """(filas con al menos un cruce, violaciones totales)."""
    dif = np.diff(Q, axis=1)
    return int((dif < 0).any(axis=1).sum()), int((dif < 0).sum())


def resumen_calib(nombre: str, y, P) -> float:
    """Imprime cobertura por tau y devuelve el ECE."""
    errs = []
    for i, t in enumerate(TAUS):
        cob = float((y <= P[:, i]).mean())
        errs.append(abs(cob - t))
    ece = float(np.mean(errs))
    print(f"   {nombre:<26} ECE={ece:.4f}   " +
          " ".join(f"{float((y <= P[:, i]).mean()):.3f}" for i in range(len(TAUS))))
    return ece


# ---------------------------------------------------------------------------
# Objetivos arctan: el de step005 (1-D) y su versión vectorizada (n, 7)
# ---------------------------------------------------------------------------
def obj_single(tau, s, std_y):
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


def obj_multi(taus, s, std_y, traza=None):
    taus_arr = np.asarray(taus, dtype=float).reshape(1, -1)
    _scale   = np.pi * (s ** 2 + std_y ** 2) ** 2 / (2.0 * s ** 3)

    def f(y_pred, dtrain):
        if traza is not None and "predt" not in traza:
            traza["predt"] = np.shape(y_pred)
            traza["label"] = np.shape(dtrain.get_label())
        y_pred = np.clip(np.asarray(y_pred, dtype=float), -1e15, 1e15)
        y      = dtrain.get_label().reshape(y_pred.shape)
        u      = y - y_pred
        s2u2   = s ** 2 + u ** 2
        grad   = -((taus_arr - 0.5 + np.arctan(u / s) / np.pi)
                   + u * s / (np.pi * s2u2)) * _scale
        hess   = ((s ** 2 + std_y ** 2) / s2u2) ** 2
        if traza is not None:
            traza["grad"] = np.shape(grad)
        return grad, hess
    return f


PARAMS = {
    "tree_method": "hist", "max_bin": 64, "max_depth": 3,
    "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 2.0, "nthread": 1, "seed": SEED,
}


def arctan_separados(X, y, s, std_y):
    d = xgb.DMatrix(X, label=y)
    return [xgb.train(PARAMS, d, num_boost_round=N_ROUNDS,
                      obj=obj_single(t, s, std_y), verbose_eval=False)
            for t in TAUS]


def nativo(X, y, estrategia: str):
    """reg:quantileerror con lista de quantile_alpha."""
    m = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.array(TAUS, dtype=float),
        multi_strategy=estrategia,
        n_estimators=N_ROUNDS,
        **{k: v for k, v in PARAMS.items() if k not in ("nthread", "seed")},
        n_jobs=1, random_state=SEED,
    )
    m.fit(X, y)
    return m


# ===========================================================================
def main() -> None:
    print(f"xgboost {xgb.__version__}   |   numpy {np.__version__}")
    print(f"config: n_train={N_TRAIN}  n_val={N_VAL}  n_feat={N_FEAT}  "
          f"rondas={N_ROUNDS}  taus={len(TAUS)}")
    rng = np.random.default_rng(SEED)

    X,  y,  _ = generar(N_TRAIN, 1.0, rng)
    Xv, yv, _ = generar(N_VAL,   1.0, rng)
    std_y = max(float(np.std(y)), 1.0)
    s     = S_FACTOR * std_y

    # ── T1 ────────────────────────────────────────────────────────────────
    _titulo("T1 · Versión de XGBoost")
    ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    R["T1"] = ver >= (2, 0)
    print(f"{_OK if R['T1'] else _FAIL} {xgb.__version__} "
          f"{'>= 2.0' if R['T1'] else '< 2.0 — multi_strategy no existe'}")
    if not R["T1"]:
        return

    # ── T2 · RUTA A: objetivo custom + multi_output_tree ──────────────────
    _titulo("T2 · RUTA A — arctan custom + multi_strategy='multi_output_tree'")
    traza: dict = {}
    try:
        Y = np.tile(y.reshape(-1, 1), (1, len(TAUS)))
        d = xgb.DMatrix(X, label=Y)
        bst = xgb.train(dict(PARAMS, multi_strategy="multi_output_tree"), d,
                        num_boost_round=N_ROUNDS,
                        obj=obj_multi(TAUS, s, std_y, traza), verbose_eval=False)
        P = bst.predict(xgb.DMatrix(Xv))
        print(f"   shapes vistos por el objetivo: {traza}")
        print(f"   predict() → {P.shape}")
        R["T2"] = P.shape == (N_VAL, len(TAUS))
        print(f"{_OK if R['T2'] else _FAIL} RUTA A "
              f"{'operativa' if R['T2'] else 'shape inesperado'}")
    except Exception as e:
        print(f"   shapes vistos por el objetivo antes de fallar: {traza}")
        print(f"{_FAIL} RUTA A bloqueada: {type(e).__name__}: {e}")
        if "size-1 arrays" in str(e):
            print("        → Booster.boost() aplana grad con c_array (solo 1-D).")
            print("          El soporte de gradientes 2-D en objetivos custom")
            print("          no está en esta versión. Requiere XGBoost >= 2.1.")
        R["T2"] = False

    # ── T3 · ¿multi_output_tree funciona con objetivo built-in? ───────────
    _titulo("T3 · Control — multi_output_tree con objetivo built-in")
    try:
        Y2 = np.column_stack([y, y * 2.0])
        m2 = xgb.XGBRegressor(multi_strategy="multi_output_tree",
                              n_estimators=20, max_depth=3, n_jobs=1)
        m2.fit(X, Y2)
        p2 = m2.predict(Xv)
        R["T3"] = p2.shape == (N_VAL, 2)
        print(f"   predict() → {p2.shape}")
        print(f"{_OK if R['T3'] else _FAIL} multi_output_tree "
              f"{'sí funciona con built-in' if R['T3'] else 'tampoco con built-in'}")
    except Exception as e:
        print(f"{_FAIL} {type(e).__name__}: {e}")
        R["T3"] = False

    # ── T4 · RUTA C: reg:quantileerror, un árbol por salida (control) ─────
    _titulo("T4 · RUTA C — reg:quantileerror multi-alpha, one_output_per_tree")
    try:
        mc = nativo(X, y, "one_output_per_tree")
        Pc = mc.predict(Xv)
        R["T4"] = Pc.shape == (N_VAL, len(TAUS))
        print(f"   predict() → {Pc.shape}")
        print(f"{_OK if R['T4'] else _FAIL} RUTA C "
              f"{'operativa' if R['T4'] else 'no disponible'}")
    except Exception as e:
        print(f"{_FAIL} {type(e).__name__}: {e}")
        R["T4"] = False
        Pc = None

    # ── T5 · RUTA B: reg:quantileerror + splits compartidos ───────────────
    _titulo("T5 · RUTA B — reg:quantileerror multi-alpha + multi_output_tree")
    try:
        mb = nativo(X, y, "multi_output_tree")
        Pb = mb.predict(Xv)
        R["T5"] = Pb.shape == (N_VAL, len(TAUS))
        print(f"   predict() → {Pb.shape}")
        print(f"{_OK if R['T5'] else _FAIL} RUTA B "
              f"{'operativa — splits compartidos' if R['T5'] else 'no disponible'}")
    except Exception as e:
        print(f"{_FAIL} {type(e).__name__}: {e}")
        R["T5"] = False
        Pb = None

    # ── T6 · comparación: cruces y calibración ────────────────────────────
    _titulo("T6 · Cruces y calibración (VAL, cuantiles verdaderos conocidos)")
    seps = arctan_separados(X, y, s, std_y)
    Pa   = np.column_stack([b.predict(xgb.DMatrix(Xv)) for b in seps])

    print(f"   {'':26} {'ECE':>5}   coberturas por τ "
          f"({' '.join(f'{t:.2f}' for t in TAUS)})")
    resumen_calib("arctan × 7 (step005 hoy)", yv, Pa)
    if Pc is not None:
        resumen_calib("RUTA C · árbol por salida", yv, Pc)
    if Pb is not None:
        resumen_calib("RUTA B · splits compartidos", yv, Pb)

    print(f"\n   {'':26} {'filas con cruce':>16} {'violaciones':>12}")
    fa, va = cruces(Pa)
    print(f"   {'arctan × 7 (step005 hoy)':<26} {fa:>10}/{N_VAL} {va:>12}")
    if Pc is not None:
        fc, vc = cruces(Pc)
        print(f"   {'RUTA C · árbol por salida':<26} {fc:>10}/{N_VAL} {vc:>12}")
    if Pb is not None:
        fb, vb = cruces(Pb)
        print(f"   {'RUTA B · splits compartidos':<26} {fb:>10}/{N_VAL} {vb:>12}")
        if fa > 0:
            print(f"\n   reducción de cruces RUTA B vs hoy: "
                  f"{100 * (1 - fb / max(fa, 1)):.0f}%")
        R["T6"] = fb <= fa

    # ── T7 · escala real del target ───────────────────────────────────────
    _titulo("T7 · Escala real del target (~1e9)")
    try:
        Xb, yb, _ = generar(N_TRAIN, 5e8, rng)
        if R.get("T5"):
            mm = nativo(Xb, yb, "multi_output_tree")
            Pm = mm.predict(Xb)
            fin = bool(np.isfinite(Pm).all())
            print(f"   RUTA B finitas={fin}  rango pred=[{np.nanmin(Pm):.3e}, "
                  f"{np.nanmax(Pm):.3e}]  target=[{yb.min():.3e}, {yb.max():.3e}]")
            R["T7"] = fin
            print(f"{_OK if fin else _FAIL} "
                  f"{'sin overflow' if fin else 'overflow / NaN'}")
        else:
            print(f"{_WARN} RUTA B no disponible — se omite")
            R["T7"] = None
    except Exception as e:
        print(f"{_FAIL} {type(e).__name__}: {e}")
        R["T7"] = False

    # ── T8 · tiempo ───────────────────────────────────────────────────────
    _titulo("T8 · Tiempo de entrenamiento")
    t0 = time.time(); arctan_separados(X, y, s, std_y); t_a = time.time() - t0
    print(f"   arctan × 7 (step005 hoy)   : {t_a:.2f} s")
    if R.get("T5"):
        t0 = time.time(); nativo(X, y, "multi_output_tree"); t_b = time.time() - t0
        print(f"   RUTA B · splits compartidos: {t_b:.2f} s   "
              f"(speedup {t_a / max(t_b, 1e-9):.1f}x)")

    # ── Veredicto ─────────────────────────────────────────────────────────
    _titulo("VEREDICTO")
    print(f"   RUTA A · arctan + multi_output_tree : "
          f"{'VIABLE' if R.get('T2') else 'BLOQUEADA (requiere XGBoost >= 2.1)'}")
    print(f"   RUTA B · reg:quantileerror + shared : "
          f"{'VIABLE' if R.get('T5') else 'no disponible'}")
    print(f"   RUTA C · reg:quantileerror por árbol: "
          f"{'VIABLE' if R.get('T4') else 'no disponible'}")
    print()
    if R.get("T2"):
        print("  → Migrar a RUTA A: conserva arctan y gana splits compartidos.")
    elif R.get("T5"):
        print("  → RUTA B es la opción realista sin actualizar XGBoost.")
        print("    Se pierde el suavizado arctan pero se ganan los splits")
        print("    compartidos, que es lo que ataca el ruido de las colas.")
        print("    Decisión: ¿actualizar a XGBoost >= 2.1 para tener ambos,")
        print("    o aceptar RUTA B? Actualizar cambia el comportamiento de")
        print("    todo el pipeline e invalida la comparación con las corridas")
        print("    anteriores.")
    else:
        print("  → Ninguna ruta multi-output disponible. Quedan: actualizar")
        print("    XGBoost, o la implementación de referencia del paper en")
        print("    github.com/LaurensSluyterman/XGBoost_quantile_regression")


if __name__ == "__main__":
    main()
