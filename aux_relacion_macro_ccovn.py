# -*- coding: utf-8 -*-
"""
aux_relacion_macro_ccovn.py
===========================
¿CCOVN y macro aportan algo para pronosticar el flujo de una entidad, MÁS ALLÁ
del calendario?

Nace de un diagnóstico sobre CREDITO: el patrón de cierre de mes que gobierna a
BBVA y al SISTEMA no existe para ese banco. Medido sobre la serie diaria
(2019-2026), el calendario da R² fuera de muestra de -6.3% para CREDITO —peor
que predecir con la media— contra +25.0% para BBVA. La pregunta natural que
sigue es si otro bloque de features cubre ese hueco.

Qué hace
--------
Para cada entidad y cada grupo de horizontes, ajusta modelos ANIDADOS y compara
su desempeño FUERA DE MUESTRA:

    base        solo intercepto             (predecir la media)
    +CAL        calendario                  (dias_al_cierre_mes, familia *_pos…)
    +FLUJO      historia del flujo propio   (ma_flujo_*, sigma_flujo_ratio…)
    +CCOVN      saldos cta.cte + overnight  (ccovn_*, share_*, var_ccovn_*)
    +MACRO      lo que quede de macro       (CDS_PERU_5Y_frac, tc_vol_ratio…)

Lo que importa es el INCREMENTO de cada bloque sobre el anterior, no su R²
aislado. Un bloque puede correlacionar fuerte y no aportar nada si el calendario
ya explicaba lo mismo.

Tres cuidados que no son opcionales acá
---------------------------------------
1. VALIDACIÓN FUERA DE MUESTRA, siempre. Las series macro son muy persistentes
   (VIX, CDS, tipo de cambio son casi caminos aleatorios). Correlacionar una
   serie persistente con un target sobre 16 años produce significancia espuria
   con enorme facilidad: la regresión encuentra una tendencia común, no una
   relación. El corte temporal (CORTE_OOS) es la única defensa barata.

2. UNA FILA POR (fecha_t, h) NO SON OBSERVACIONES INDEPENDIENTES. Para un h
   dado, fechas consecutivas comparten casi toda su ventana de features, y para
   una fecha dada los 74 horizontes comparten el mismo origen. Por eso NO se
   reportan p-valores: darían significancia a cualquier cosa. Se reporta mejora
   fuera de muestra, que es inmune a ese problema.

3. COLUMNAS ENTERAMENTE NaN. Para un banco individual fuera de la partición,
   resolver_ccovn_lados() devuelve contraparte=None y ccovn_contraparte_lag1
   queda en NaN completo. Se descartan esas columnas y se REPORTA cuáles, en vez
   de dejar que un dropna() silencioso se lleve todas las filas.

Uso
---
Desde Spyder, sin argumentos (usa RUTA_MATRIZ_DEFECTO):

    runfile('aux_relacion_macro_ccovn.py')

O apuntando a otra matriz:

    runfile('aux_relacion_macro_ccovn.py', args='--matriz "1. Data/Clean/otra.parquet"')

Nota sobre el alcance de MACRO: step001 excluye casi todo el bloque macro vía
FEATURES_EXCLUIR (TC_PEN_USD, EMBI_PERU, delta_EMBI, VIX_frac, T10Y_frac,
diferencial_tasas, garch_vol_embi, …). Este script reporta explícitamente qué
columnas macro esperaba y no encontró, para que quede claro que un "MACRO no
aporta" puede significar "macro casi no está en la matriz". Para probarlas hay
que comentarlas en FEATURES_EXCLUIR —convención invertida: comentar = ACTIVAR—
y regenerar la matriz.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────
RUTA_MATRIZ_DEFECTO = "1. Data/Clean/matriz_features.parquet"

# Entidades a comparar. El contraste es el punto: sin una entidad donde el
# calendario SÍ funciona (BBVA), un R² bajo en CREDITO no se distingue de que
# el método esté mal aplicado.
ENTIDADES = ["CREDITO", "BBVA", "SISTEMA"]

# Corte temporal para la validación. Todo lo anterior entrena, lo posterior
# evalúa. No es aleatorio a propósito: un split aleatorio sobre series
# temporales filtra el futuro al pasado por las ventanas móviles.
CORTE_OOS = "2024-01-01"

# Los mismos grupos que usa step005 para reportar métricas.
GRUPOS_H = [(2, 5), (6, 15), (16, 30), (31, 50), (51, 75)]

# Ridge chico: varias features de calendario son casi colineales entre sí
# (dias_al_cierre_mes y la familia *_pos comparten el reloj del mes) y la
# solución por mínimos cuadrados puros queda mal condicionada.
LAMBDA_RIDGE = 1e-3


# ── Bloques de features ───────────────────────────────────────────────────────
# Se declaran por PREFIJO/nombre y se intersectan con lo que la matriz traiga:
# las versiones de step001 difieren en qué columnas sobreviven a FEATURES_EXCLUIR,
# y una lista fija dejaría el script atado a una sola de ellas.
BLOQUES = {
    "CAL": dict(
        exactos={"dias_al_cierre_mes", "dias_desde_cierre_mes", "dias_al_cierre_trim",
                 "is_cierre_encaje", "es_post_feriado", "dias_desde_ultimo_habil"},
        prefijos=("esc_", "acum_", "dias_", "mes_", "dias_sem_", "elec_"),
    ),
    "FLUJO": dict(
        exactos={"ma_flujo_5d", "ma_flujo_20d", "sigma_flujo_ratio", "R_conf_t2",
                 "flujo_neto_acum_mes"},
        prefijos=("ma_flujo", "sigma_flujo"),
    ),
    "CCOVN": dict(
        exactos=set(),
        prefijos=("ccovn_", "var_ccovn_", "share_"),
    ),
    "MACRO": dict(
        exactos={"CDS_PERU_5Y_frac", "tc_vol_ratio", "VIX", "T10Y", "FED_FUNDS",
                 "TASA_REF_BCRP", "EMBI_PERU_frac", "VIX_frac", "T10Y_frac",
                 "TC_PEN_USD", "EMBI_PERU", "diferencial_tasas"},
        prefijos=("cds_", "vix", "embi", "tc_"),
    ),
}

# Macro que step001 típicamente excluye. Solo para el reporte de cobertura: si
# faltan, "MACRO no aporta" significa "macro no está", no "macro no sirve".
MACRO_ESPERADA = ["CDS_PERU_5Y_frac", "VIX_frac", "T10Y_frac", "EMBI_PERU_frac",
                  "TC_PEN_USD", "diferencial_tasas", "tc_vol_ratio"]

IDENTIDAD = {"fecha_t", "fecha_th", "banco", "h", "target"}


def clasificar_columnas(cols):
    """Asigna cada columna numérica a un bloque. Devuelve (dict bloque->cols, sobrantes)."""
    disponibles = [c for c in cols if c not in IDENTIDAD]
    asignadas, fuera = {}, set(disponibles)
    for nombre, reglas in BLOQUES.items():
        sel = []
        for c in disponibles:
            if c in reglas["exactos"] or c.lower().startswith(reglas["prefijos"]):
                sel.append(c)
        # Un mismo nombre no puede caer en dos bloques: el primero que lo toma se
        # lo queda, y el orden de BLOQUES es el orden de anidamiento. Sin esto,
        # ccovn_* entraría también a MACRO por el prefijo "tc_"/"cds_" cruzado.
        sel = [c for c in sel if c in fuera]
        asignadas[nombre] = sorted(sel)
        fuera -= set(sel)
    return asignadas, sorted(fuera)


def _fmt_tabla(t, fmt):
    """
    DataFrame de strings formateados, sin depender de applymap ni de DataFrame.map.

    applymap salió en pandas 3.x y DataFrame.map recién entró en 2.1: usar
    cualquiera de los dos ata el script a un rango de versiones. El repo corre
    sobre la que tenga instalada Spyder, que no controlamos desde acá.
    """
    out = t.copy().astype(object)
    for c in out.columns:
        out[c] = [fmt(v) for v in t[c]]
    return out


def _ridge_oos(X_tr, y_tr, X_te, y_te):
    """
    R² fuera de muestra de una ridge estandarizada.

    Estandarizar con la media/sd del TRAIN, nunca del total: usar el sd del
    período completo mete información del futuro en el escalado. Con ridge eso
    además cambia la penalización efectiva de cada feature.
    """
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr = np.column_stack([np.ones(len(X_tr)), (X_tr - mu) / sd])
    Zte = np.column_stack([np.ones(len(X_te)), (X_te - mu) / sd])

    G = Ztr.T @ Ztr
    reg = LAMBDA_RIDGE * np.trace(G) / max(G.shape[0], 1)
    G[1:, 1:] += reg * np.eye(G.shape[0] - 1)      # no penalizar el intercepto
    beta = np.linalg.solve(G, Ztr.T @ y_tr)

    pred = Zte @ beta
    sse = float(((y_te - pred) ** 2).mean())
    # Baseline = media del TRAIN, no del test: predecir la media del test ya
    # usaría el futuro. Por eso el R² así definido puede ser negativo, y eso es
    # informativo, no un error.
    base = float(((y_te - y_tr.mean()) ** 2).mean())
    return (1 - sse / base) if base > 0 else np.nan, float(np.abs(y_te - pred).mean())


def analizar_entidad(df, entidad, bloques_cols):
    print(f"\n{'='*78}")
    print(f"  {entidad}")
    print(f"{'='*78}")

    d = df[df["banco"] == entidad]
    if d.empty:
        print(f"  [AVISO] '{entidad}' no está en la matriz — omitida")
        return None

    # Columnas enteramente NaN para ESTA entidad (típico: ccovn_contraparte_lag1
    # en un banco fuera de la partición). Se sacan acá y no antes: qué está vacío
    # depende de la entidad, no de la matriz.
    vacias = {}
    bl = {}
    for nombre, cols in bloques_cols.items():
        usables = [c for c in cols if d[c].notna().any()]
        nulas = [c for c in cols if c not in usables]
        if nulas:
            vacias[nombre] = nulas
        bl[nombre] = usables
    if vacias:
        for nombre, cols in vacias.items():
            print(f"  [nota] {nombre}: {len(cols)} columna(s) vacías para esta "
                  f"entidad, descartadas: {cols}")

    filas = []
    for h_lo, h_hi in GRUPOS_H:
        sub = d[(d["h"] >= h_lo) & (d["h"] <= h_hi)]
        acum = []
        fila = {"grupo": f"h{h_lo:02d}-{h_hi:02d}", "n_train": 0, "n_test": 0}
        for nombre in ["CAL", "FLUJO", "CCOVN", "MACRO"]:
            acum = acum + bl.get(nombre, [])
            if not acum:
                fila[nombre] = np.nan
                continue
            s = sub[["fecha_t", "target"] + acum].dropna()
            tr = s[s["fecha_t"] < CORTE_OOS]
            te = s[s["fecha_t"] >= CORTE_OOS]
            # Se registran las n del último bloque que sí corrió: el dropna las
            # cambia entre bloques (una columna con NaN se lleva filas enteras),
            # así que reportar una sola cifra por grupo sería engañoso si
            # difieren mucho. Se avisa cuando difieren.
            if len(tr) < 200 or len(te) < 100:
                fila[nombre] = np.nan
                continue
            r2, _ = _ridge_oos(tr[acum].to_numpy(float), tr["target"].to_numpy(float),
                               te[acum].to_numpy(float), te["target"].to_numpy(float))
            fila[nombre] = r2
            if fila["n_train"] and abs(len(tr) - fila["n_train"]) > 0.05 * fila["n_train"]:
                fila["aviso_n"] = True
            fila["n_train"], fila["n_test"] = len(tr), len(te)
        filas.append(fila)

    t = pd.DataFrame(filas).set_index("grupo")
    print(f"\n  R² FUERA DE MUESTRA, modelos anidados (train < {CORTE_OOS} ≤ test)")
    print(f"  Cada columna AGREGA su bloque al anterior. Negativo = peor que la media.\n")
    cols_r2 = [c for c in ["CAL", "FLUJO", "CCOVN", "MACRO"] if c in t.columns]
    vista = t[cols_r2].copy()
    print("   " + _fmt_tabla(vista,
          lambda v: f"{v:7.2%}" if pd.notna(v) else "      —").to_string())
    print(f"\n   n train/test por grupo: "
          + ", ".join(f"{g}: {int(r['n_train']):,}/{int(r['n_test']):,}"
                      for g, r in t.iterrows()))
    if t.get("aviso_n", pd.Series(dtype=bool)).any():
        print("   [nota] el nº de filas cambió >5% entre bloques en algún grupo: "
              "una columna con NaN se llevó filas enteras al hacer dropna, así "
              "que esas comparaciones no son sobre exactamente la misma muestra.")

    print(f"\n  APORTE INCREMENTAL de cada bloque (puntos porcentuales de R²)\n")
    inc = vista.copy()
    for i, c in enumerate(cols_r2):
        inc[c] = (vista[c] - vista[cols_r2[i - 1]]) * 100 if i else vista[c] * 100
    print("   " + _fmt_tabla(inc,
          lambda v: f"{v:+7.2f}" if pd.notna(v) else "      —").to_string())
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matriz", metavar="RUTA", default=None)
    ap.add_argument("--entidades", metavar="A,B,C", default=None,
                    help="lista separada por comas; por defecto CREDITO,BBVA,SISTEMA")
    a = ap.parse_args()

    ruta = a.matriz or RUTA_MATRIZ_DEFECTO
    if not os.path.exists(ruta):
        print(f"[ERROR] no existe la matriz: {ruta}")
        print("        Ajustar RUTA_MATRIZ_DEFECTO o pasar --matriz RUTA")
        sys.exit(1)
    entidades = ([e.strip() for e in a.entidades.split(",")] if a.entidades
                 else ENTIDADES)

    print("=" * 78)
    print("  ¿APORTAN CCOVN Y MACRO MÁS ALLÁ DEL CALENDARIO?")
    print("=" * 78)

    df = pd.read_parquet(ruta)
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    print(f"\nMatriz: {Path(ruta).name} — {len(df):,} filas, {df.shape[1]} columnas")
    print(f"Corte fuera de muestra: {CORTE_OOS}")

    num = [c for c in df.columns
           if c not in IDENTIDAD and pd.api.types.is_numeric_dtype(df[c])]
    bloques_cols, sobrantes = clasificar_columnas(num)

    print("\nBloques de features detectados en esta matriz:")
    for nombre, cols in bloques_cols.items():
        print(f"  {nombre:6s} ({len(cols):2d}): {cols if cols else '— ninguna —'}")
    if sobrantes:
        print(f"  {'(sin clasificar)':6s} ({len(sobrantes):2d}): {sobrantes}")

    faltan = [c for c in MACRO_ESPERADA if c not in num]
    if faltan:
        print(f"\n  [IMPORTANTE] macro esperada y AUSENTE de la matriz ({len(faltan)}):")
        print(f"    {faltan}")
        print("    Están en FEATURES_EXCLUIR de step001. Un 'MACRO no aporta' de")
        print("    abajo se refiere SOLO a lo que sí está. Para probar las demás:")
        print("    comentarlas en FEATURES_EXCLUIR (comentar = ACTIVAR, convención")
        print("    invertida) y regenerar la matriz.")

    resultados = {}
    for e in entidades:
        r = analizar_entidad(df, e, bloques_cols)
        if r is not None:
            resultados[e] = r

    print(f"\n{'='*78}")
    print("  CÓMO LEER ESTO")
    print(f"{'='*78}")
    print("  · El R² fuera de muestra puede ser NEGATIVO: significa que el modelo")
    print("    predice peor que la media del período de entrenamiento. Con series")
    print("    macro persistentes es el resultado más común y NO es un bug.")
    print("  · Un aporte incremental de +0.1 pp no es señal: es ruido de muestreo.")
    print("    Lo que vale la pena mirar son saltos de varios puntos, y que se")
    print("    repitan en varios grupos de horizonte.")
    print("  · Si un bloque aporta en una entidad y no en otra, eso ES el hallazgo:")
    print("    justifica modelar esas entidades por separado.")


if __name__ == "__main__":
    main()
