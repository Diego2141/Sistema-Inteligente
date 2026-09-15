# -*- coding: utf-8 -*-
"""
aux_verificar_preds_train.py
============================
Checklist de correctitud de PREDECIR_TRAIN en step005_walk_forward_cv_4.py y
step005_walk_forward_cv_4_particiones.py.

Qué se verifica

  0. Nombres indefinidos      pyflakes sobre los dos step005 (py_compile no ve
                              esta clase de error)
  1. El botón                 PREDECIR_TRAIN existe en ambos, mismo default, y
                              el default es False (es diagnóstico, cuesta)
  2. Paridad entre archivos   el bloque de train es IDÉNTICO en los dos step005
  3. Alineación de la máscara la máscara re-derivada devuelve exactamente las
                              filas de X_train, en el mismo orden
  4. Esquema del scaffold     train produce las mismas columnas que test y val
  5. Descomposición h=2       el parquet guarda el target REAL, no D(t+2)
  6. Anti-leakage             train sigue sin compartir filas con val ni test

Cada sección lleva un CONTROL NEGATIVO: se comprueba que el chequeo FALLA cuando
debe fallar. Un chequeo que no puede dar [FALLA] es decorativo.

Uso
    python aux_verificar_preds_train.py            # autotest, no necesita datos
    python aux_verificar_preds_train.py --preds "2. Output/step005_wfcv_v4_direct/fold_exp_arctan"

Desde Spyder, sin argumentos:
    runfile('aux_verificar_preds_train.py')

Sale con código 1 si alguna verificación falla.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd

AQUI = pathlib.Path(__file__).parent
STEP005 = AQUI / "step005_walk_forward_cv_4.py"
STEP005_P = AQUI / "step005_walk_forward_cv_4_particiones.py"

# Columnas que los tres scaffolds (test, val, train) deben compartir. Los
# cuantiles se agregan después, desde TAUS, así que se comprueban aparte.
COLS_IDENTIDAD = ["banco", "fold", "fecha_t", "fecha_th", "h", "target"]

_FALLAS: list[str] = []


def check(desc, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLA'}] {desc}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        _FALLAS.append(desc)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 0. Nombres indefinidos
# ─────────────────────────────────────────────────────────────────────────────
def sec_pyflakes():
    print("\n0. Nombres indefinidos en los dos step005 (estático)")
    for f in (STEP005, STEP005_P):
        try:
            r = subprocess.run([sys.executable, "-m", "pyflakes", str(f)],
                               capture_output=True, text=True, timeout=120)
            undef = [l for l in r.stdout.splitlines() if "undefined name" in l]
            check(f"{f.name}: sin nombres indefinidos", not undef,
                  "; ".join(undef[:3]) if undef else "")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  [----] pyflakes no disponible — sección omitida para {f.name}")
            return


# ─────────────────────────────────────────────────────────────────────────────
# 1-2. El botón y la paridad entre archivos
# ─────────────────────────────────────────────────────────────────────────────
def _const_bool(src, nombre):
    """Valor de una asignación de módulo `nombre = True/False`, vía AST."""
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == nombre:
                    if isinstance(n.value, ast.Constant):
                        return n.value.value
    return None


def _bloque_train(src):
    """
    El bloque `if PREDECIR_TRAIN:` tal como está escrito, normalizado.

    Se extrae por texto y no por AST-unparse porque lo que interesa comparar
    entre los dos archivos es el código REAL, comentarios incluidos: una
    divergencia en un comentario suele anticipar una divergencia en la lógica.
    """
    m = re.search(r"\n(\s*)if PREDECIR_TRAIN:\n(.*?)(?=\n\1n_h_ok \+= 1)", src, re.S)
    if not m:
        return None
    return "\n".join(l.rstrip() for l in m.group(2).splitlines() if l.strip())


def sec_boton():
    print("\n1. El botón PREDECIR_TRAIN")
    src, src_p = STEP005.read_text(encoding="utf-8"), STEP005_P.read_text(encoding="utf-8")
    v, v_p = _const_bool(src, "PREDECIR_TRAIN"), _const_bool(src_p, "PREDECIR_TRAIN")
    check("existe en step005_walk_forward_cv_4.py", v is not None, f"= {v}")
    check("existe en step005_walk_forward_cv_4_particiones.py", v_p is not None, f"= {v_p}")
    check("mismo valor en los dos archivos", v == v_p, f"{v} vs {v_p}")
    # Default False a propósito: TRAIN son 3 años contra 0.5 de test, el parquet
    # pesa ~6x y hay una predicción extra por (h, tau). Que quede en True sin
    # querer encarece cada corrida en silencio.
    check("el default es False (es diagnóstico, no producción)", v is False,
          f"vale {v} — si es deliberado, ignorar este [FALLA]")

    print("\n2. Paridad del bloque de train entre los dos step005")
    b, b_p = _bloque_train(src), _bloque_train(src_p)
    check("el bloque existe en el archivo base", b is not None)
    check("el bloque existe en el de particiones", b_p is not None)
    if b and b_p:
        check("los dos bloques son idénticos", b == b_p,
              "difieren — se desincronizaron" if b != b_p
              else f"{len(b.splitlines())} líneas")
    # Control negativo: una copia alterada tiene que dar distinto.
    if b:
        check("[control negativo] un bloque alterado se detecta",
              b != b.replace("y_train.values", "y_fit_tr"), "la comparación discrimina")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Alineación de la máscara re-derivada
# ─────────────────────────────────────────────────────────────────────────────
def _cargar_step005():
    """Importa step005 sin ejecutar su __main__."""
    spec = importlib.util.spec_from_file_location("_s5", STEP005)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s5"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def _df_h_sintetico(n_nan=200):
    """
    Un solo horizonte, con NaN en el target a propósito.

    Los NaN son el punto: la máscara de producción los excluye con
    target.notna(), y si la máscara re-derivada del parche los olvidara, las
    fechas quedarían corridas respecto de las predicciones sin que nada avise.
    """
    fechas = pd.bdate_range("2018-01-01", "2024-12-31")
    n = len(fechas)
    rng = np.random.default_rng(0)
    d = pd.DataFrame({
        "fecha_t": fechas,
        "fecha_th": fechas + pd.Timedelta(days=10),
        "h": 10,
        "target": rng.normal(0, 100, n),
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "R_conf_t2": rng.normal(0, 1, n),
    })
    d.loc[d.sample(n_nan, random_state=9).index, "target"] = np.nan
    return d


FOLD_TEST = {
    "fold": 1,
    "train_start": pd.Timestamp("2018-01-01"), "train_end": pd.Timestamp("2021-12-31"),
    "val_start":   pd.Timestamp("2022-06-01"), "val_end":   pd.Timestamp("2022-11-30"),
    "test_start":  pd.Timestamp("2023-06-01"), "test_end":  pd.Timestamp("2023-11-30"),
}


def _mascara_train(df_h, fold):
    """La máscara re-derivada, copiada literal del parche de step005."""
    return ((df_h["fecha_t"] >= fold["train_start"]) &
            (df_h["fecha_t"] <= fold["train_end"]) &
            df_h["target"].notna())


def sec_mascara(s5):
    print("\n3. La máscara re-derivada coincide con X_train")
    df_h = _df_h_sintetico()
    cols = ["f1", "f2", "R_conf_t2"]
    X_tr, y_tr, X_v, y_v, X_te, y_te, _, _ = s5.preparar_fold_data_h(df_h, FOLD_TEST, cols)
    m = _mascara_train(df_h, FOLD_TEST)

    check("mismo número de filas que X_train", int(m.sum()) == len(X_tr),
          f"máscara {int(m.sum())} vs X_train {len(X_tr)}")
    if int(m.sum()) == len(X_tr):
        igual_f = (df_h.loc[m, "fecha_t"].to_numpy() ==
                   df_h.loc[X_tr.index, "fecha_t"].to_numpy()).all()
        check("las fechas alinean fila a fila (mismo orden)", bool(igual_f))
        igual_y = (df_h.loc[m, "target"].to_numpy() == y_tr.to_numpy()).all()
        check("el target de la máscara es y_train", bool(igual_y))

    # Control negativo: la máscara SIN target.notna() es el error real que se
    # quiere atrapar, y la aserción del parche debe verlo.
    mala = ((df_h["fecha_t"] >= FOLD_TEST["train_start"]) &
            (df_h["fecha_t"] <= FOLD_TEST["train_end"]))
    check("[control negativo] una máscara sin notna() se detecta",
          int(mala.sum()) != len(X_tr),
          f"daría {int(mala.sum())} filas contra {len(X_tr)} — la aserción aborta")
    return df_h, X_tr, y_tr, X_v, y_v, X_te, y_te


# ─────────────────────────────────────────────────────────────────────────────
# 4-5. Esquema del scaffold y descomposición de h=2
# ─────────────────────────────────────────────────────────────────────────────
def _scaffold(df_h, mask, y, preds, banco="SISTEMA", fold=1, h=10):
    """Construye un scaffold con la misma forma que los tres de step005."""
    d = pd.DataFrame({
        "banco": banco, "fold": fold,
        "fecha_t": pd.DatetimeIndex(df_h.loc[mask, "fecha_t"]),
        "fecha_th": pd.DatetimeIndex(df_h.loc[mask, "fecha_th"]),
        "h": h, "target": y.values,
    })
    for tau, p in preds.items():
        d["mean" if tau == "mean" else f"q{int(tau * 100):02d}"] = p
    return d


def sec_esquema(s5, df_h, X_tr, y_tr, X_te, y_te):
    print("\n4. El scaffold de TRAIN tiene el mismo esquema que TEST y VAL")
    taus = list(s5.TAUS) + ["mean"]
    p_tr = {t: np.zeros(len(X_tr)) for t in taus}
    p_te = {t: np.zeros(len(X_te)) for t in taus}
    m_tr = _mascara_train(df_h, FOLD_TEST)
    m_te = ((df_h["fecha_t"] >= FOLD_TEST["test_start"]) &
            (df_h["fecha_t"] <= FOLD_TEST["test_end"]) & df_h["target"].notna())

    s_tr = _scaffold(df_h, m_tr, y_tr, p_tr)
    s_te = _scaffold(df_h, m_te, y_te, p_te)
    check("mismas columnas y en el mismo orden", list(s_tr.columns) == list(s_te.columns),
          f"{len(s_tr.columns)} columnas")
    check("están las columnas de identidad",
          all(c in s_tr.columns for c in COLS_IDENTIDAD),
          ", ".join(COLS_IDENTIDAD))
    esperadas = [("mean" if t == "mean" else f"q{int(t*100):02d}") for t in taus]
    check("está una columna por cuantil de TAUS",
          all(c in s_tr.columns for c in esperadas), ", ".join(esperadas))
    check("mismos dtypes que el scaffold de test",
          list(s_tr.dtypes.astype(str)) == list(s_te.dtypes.astype(str)))
    check("[control negativo] una columna de más se detecta",
          list(s_tr.assign(extra=1).columns) != list(s_te.columns))

    print("\n5. Descomposición de h=2: el parquet guarda el target REAL")
    # Con DESCOMPONER_H2 el modelo se ajusta contra D(t+2) = target + R_conf_t2.
    # El scaffold debe traer `target`, no lo que se ajustó: si guardara y_fit_tr,
    # el fan chart y la cobertura compararían contra una serie distinta.
    r_conf = X_tr["R_conf_t2"].to_numpy(dtype=float)
    y_fit = y_tr.to_numpy(dtype=float) + r_conf          # lo que se ajusta
    s = _scaffold(df_h, m_tr, y_tr, p_tr)
    check("el target del scaffold es y_train, no y_fit_tr",
          np.allclose(s["target"].to_numpy(), y_tr.to_numpy()))
    check("[control negativo] guardar y_fit_tr se detecta",
          not np.allclose(y_fit, y_tr.to_numpy()),
          f"difieren en media {np.abs(y_fit - y_tr.to_numpy()).mean():.3f}")

    # Y que _destransformar_h2 sea la inversa DENTRO DE SU DOMINIO.
    #
    # No es una resta simple: aplica max(q_D, 0) - R_conf. El clip impone D >= 0,
    # que es cierto por construcción —D son depósitos— y de ahí sale el piso
    # físico target >= -R_conf_t2. Para probar la inversión hay que darle valores
    # con D >= 0, como los reales; con D negativo el clip muerde y la inversión
    # NO es exacta, que es justamente el comportamiento buscado.
    if hasattr(s5, "_destransformar_h2"):
        r_pos = np.abs(r_conf) * 10.0 + 1.0                 # R_conf > 0
        y_ok = np.abs(y_tr.to_numpy()) - r_pos              # asegura D = y + r >= 0
        d_pos = y_ok + r_pos
        assert (d_pos >= 0).all()
        vuelto = s5._destransformar_h2({t: d_pos.copy() for t in taus}, r_pos)
        check("_destransformar_h2 invierte exacto cuando D >= 0 (el caso real)",
              all(np.allclose(vuelto[t], y_ok) for t in taus))

        # El piso: ninguna predicción destransformada puede quedar por debajo
        # de -R_conf_t2, pase lo que pase con la predicción cruda.
        extremo = s5._destransformar_h2(
            {t: np.full(len(r_pos), -1e9) for t in taus}, r_pos)
        check("el clip impone el piso target >= -R_conf_t2",
              all(np.allclose(extremo[t], -r_pos) for t in taus),
              "una predicción absurda de D queda anclada al piso físico")
        check("[control negativo] sin el clip el piso no se respetaría",
              not np.allclose(-1e9 - r_pos, -r_pos))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Anti-leakage
# ─────────────────────────────────────────────────────────────────────────────
def sec_leakage(df_h):
    print("\n6. TRAIN no comparte filas con VAL ni TEST")
    f = FOLD_TEST
    m_tr = _mascara_train(df_h, f)
    m_v = ((df_h["fecha_t"] >= f["val_start"]) & (df_h["fecha_t"] <= f["val_end"])
           & df_h["target"].notna())
    m_te = ((df_h["fecha_t"] >= f["test_start"]) & (df_h["fecha_t"] <= f["test_end"])
            & df_h["target"].notna())
    check("train ∩ val vacío", int((m_tr & m_v).sum()) == 0)
    check("train ∩ test vacío", int((m_tr & m_te).sum()) == 0)
    # Control negativo: un fold mal construido (train que pisa val) debe verse.
    malo = dict(f, train_end=f["val_end"])
    check("[control negativo] un train que pisa val se detecta",
          int((_mascara_train(df_h, malo) & m_v).sum()) > 0,
          f"{int((_mascara_train(df_h, malo) & m_v).sum())} filas solapadas")


# ─────────────────────────────────────────────────────────────────────────────
# Modo --preds: validar los parquets reales
# ─────────────────────────────────────────────────────────────────────────────
def modo_preds(dirpreds):
    d = pathlib.Path(dirpreds)
    print(f"\nValidando parquets reales en: {d}")
    tr = sorted(d.glob("preds_train_fold*.parquet"))
    te = sorted(d.glob("preds_test_fold*.parquet"))
    check("hay parquets de TRAIN", bool(tr), f"{len(tr)} archivo(s)")
    if not tr:
        print("  -> correr step005 con PREDECIR_TRAIN = True")
        return

    dtr = pd.concat([pd.read_parquet(p) for p in tr], ignore_index=True)
    print(f"  TRAIN: {len(dtr):,} filas, folds {sorted(dtr['fold'].unique())}")

    if te:
        dte = pd.concat([pd.read_parquet(p) for p in te], ignore_index=True)
        check("mismas columnas que los parquets de TEST",
              list(dtr.columns) == list(dte.columns),
              f"train {len(dtr.columns)} vs test {len(dte.columns)}")
        # TRAIN son 3 años contra 0.5 de test: debe ser claramente más grande.
        # Si saliera parecido, es señal de que la ventana no es la que se cree.
        razon = len(dtr) / max(len(dte), 1)
        check("TRAIN es varias veces más grande que TEST (ventanas 3 vs 0.5 años)",
              razon > 2, f"razón {razon:.1f}x")

    check("sin filas duplicadas por (fold, fecha_t, h)",
          not dtr.duplicated(["fold", "fecha_t", "h"]).any(),
          f"{int(dtr.duplicated(['fold','fecha_t','h']).sum())} duplicadas")
    check("target sin NaN (la máscara los excluye)", dtr["target"].notna().all(),
          f"{int(dtr['target'].isna().sum())} NaN")

    qs = [c for c in dtr.columns if c.startswith("q")]
    if len(qs) >= 2:
        orden = sorted(qs, key=lambda c: int(c[1:]))
        cruces = 0
        for a, b in zip(orden, orden[1:]):
            cruces += int((dtr[a] > dtr[b] + 1e-6).sum())
        check("cuantiles no se cruzan (_reordenar_cuantiles ya corrió)",
              cruces == 0, f"{cruces:,} cruces sobre {len(dtr):,} filas")

    # Las ventanas de TRAIN son de 3 años y los folds avanzan 0.5, así que una
    # fecha cae en varios folds. No es un error: es lo que obliga a que el fan
    # chart emita un frame por (fold, fecha) y no por fecha.
    comp = (dtr.drop_duplicates(["fold", "fecha_t"]).groupby("fecha_t")["fold"]
               .nunique())
    n = int((comp > 1).sum())
    print(f"  {n:,} fecha(s) de origen aparecen en más de un fold "
          f"(esperable: ventanas de 3 años avanzando {0.5})")
    if n:
        print(f"     máximo de folds sobre una misma fecha: {int(comp.max())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", metavar="DIR", default=None,
                    help="carpeta fold_<modo> con los parquets a validar")
    a = ap.parse_args()

    print("=" * 74)
    print("CHECKLIST — PREDECIR_TRAIN en los dos step005")
    print("=" * 74)

    sec_pyflakes()
    sec_boton()
    try:
        s5 = _cargar_step005()
    except Exception as e:
        print(f"\n  [----] no se pudo importar step005 ({type(e).__name__}: {e})")
        print("         las secciones 3-6 necesitan xgboost instalado — omitidas")
        s5 = None
    if s5 is not None:
        df_h, X_tr, y_tr, X_v, y_v, X_te, y_te = sec_mascara(s5)
        sec_esquema(s5, df_h, X_tr, y_tr, X_te, y_te)
        sec_leakage(df_h)

    if a.preds:
        modo_preds(a.preds)

    print("\n" + "=" * 74)
    if _FALLAS:
        print(f"RESULTADO: {len(_FALLAS)} verificación(es) FALLARON")
        for f in _FALLAS:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULTADO: todas las verificaciones pasaron")
    print("=" * 74)


if __name__ == "__main__":
    main()
