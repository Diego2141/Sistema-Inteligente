# -*- coding: utf-8 -*-
"""
aux_fanchart_horizontes.py
Fan chart del modelo a través de todos los horizontes h = 1..90.

Para cada fecha t del período TEST, traza la ruta realizada de D-R
desde h=1 hasta h=90, superpuesta sobre las bandas de predicción del
modelo (Q01-Q99, Q05-Q95) agregadas sobre todas las fechas del test.

Permite ver:
  - Cómo crece la incertidumbre con el horizonte (term structure).
  - Si el modelo abre sus bandas apropiadamente a medida que h aumenta.
  - Qué rutas realizadas quedan fuera de las bandas predichas.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO        = "SISTEMA"
SEMANAS_VAL  = 26
SEMANAS_TEST = 13
QUANTILES    = [0.01, 0.05, 0.50, 0.95, 0.99]


# ── 1. Cargar modelo más reciente ─────────────────────────────────────────
def cargar_modelos(banco: str, dir_modelos: Path) -> tuple[dict, list[str]]:
    metas = sorted(dir_modelos.glob(f"metadata_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(f"No se encontró metadata para banco={banco}")
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        patron = f"lgbm_{banco}_q{int(tau*100):02d}_{fecha}.txt"
        ruta   = dir_modelos / patron
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        modelos[tau] = lgb.Booster(model_file=str(ruta))

    print(f"Modelos cargados: {metas[0].name}")
    print(f"  Quantiles : {quantiles}")
    print(f"  Features  : {len(cols_feat)}")
    return modelos, cols_feat


# ── 2. Leer datos y aplicar split ────────────────────────────────────────────
def leer_y_split(banco: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    fechas = np.sort(df["fecha_t"].unique())
    n = len(fechas)
    n_test = min(SEMANAS_TEST * 5, n // 6)
    n_val  = min(SEMANAS_VAL  * 5, n // 4)
    corte_val  = fechas[n - n_test - n_val]
    corte_test = fechas[n - n_test]

    df_train = df[df["fecha_t"] <  corte_val].copy()
    df_test  = df[df["fecha_t"] >= corte_test].copy()

    # Imputar con mediana de train (sin leak)
    cols_excluir = {"fecha_t", "banco", "target"}
    cols_feat = [c for c in df.columns if c not in cols_excluir]
    medianas = df_train[cols_feat].median()
    df_test[cols_feat] = df_test[cols_feat].fillna(medianas)

    print(f"\nTEST: {df_test['fecha_t'].min().date()} → {df_test['fecha_t'].max().date()}")
    print(f"  {df_test['fecha_t'].nunique()} fechas × {df_test['h'].nunique()} horizontes")
    return df_train, df_test


# ── 3. Generar predicciones ───────────────────────────────────────────────────
def predecir(modelos: dict, df: pd.DataFrame, cols_feat: list[str]) -> pd.DataFrame:
    X = df[cols_feat].copy()
    preds = df[["fecha_t", "h", "target"]].copy()
    for tau, model in modelos.items():
        preds[f"q{int(tau*100):02d}"] = model.predict(X)
    # Corrección de monotonicidad
    q_cols = sorted([c for c in preds.columns if c.startswith("q")])
    preds[q_cols] = np.sort(preds[q_cols].values, axis=1)
    return preds


# ── 4. Agregar por horizonte h ────────────────────────────────────────────────
def agregar_por_h(preds: pd.DataFrame) -> pd.DataFrame:
    q_cols = [c for c in preds.columns if c.startswith("q")]
    agg = preds.groupby("h")[q_cols + ["target"]].median()
    return agg


# ── 5. Graficar ───────────────────────────────────────────────────────────────
def graficar(preds: pd.DataFrame, banco: str):
    agg   = agregar_por_h(preds)
    hs    = agg.index.values
    fechas_test = sorted(preds["fecha_t"].unique())
    n_fechas    = len(fechas_test)

    # Colores por fecha t
    cmap    = cm.get_cmap("Blues", n_fechas + 4)
    colores = [cmap(i + 3) for i in range(n_fechas)]

    fig, ax = plt.subplots(figsize=(16, 7))

    # ── Bandas del modelo (medianas sobre fechas del test) ────────────────
    ax.fill_between(hs, agg["q01"] / 1e6, agg["q99"] / 1e6,
                    alpha=0.15, color="steelblue", label="Q01–Q99 (mediana test)")
    ax.fill_between(hs, agg["q05"] / 1e6, agg["q95"] / 1e6,
                    alpha=0.30, color="steelblue", label="Q05–Q95 (mediana test)")
    ax.plot(hs, agg["q50"] / 1e6,
            color="steelblue", lw=2, label="Mediana predicha (Q50)")

    # ── Ruta realizada por fecha t ────────────────────────────────────────
    for i, fecha in enumerate(fechas_test):
        sub = (preds[preds["fecha_t"] == fecha]
               .sort_values("h")
               .dropna(subset=["target"]))
        if sub.empty:
            continue
        ax.plot(sub["h"], sub["target"] / 1e6,
                color=colores[i], lw=1.0, alpha=0.55, zorder=3)

    # Línea de referencia y decoración
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)

    # Colorbar como leyenda de fechas
    sm = plt.cm.ScalarMappable(
        cmap=cm.get_cmap("Blues"),
        norm=plt.Normalize(0, n_fechas)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Fecha t en TEST (más oscuro = más reciente)", fontsize=9)
    n_ticks = min(6, n_fechas)
    tick_idx = np.linspace(0, n_fechas - 1, n_ticks, dtype=int)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels(
        [pd.Timestamp(fechas_test[i]).strftime("%d %b %Y") for i in tick_idx],
        fontsize=8
    )

    ax.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)
    ax.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax.set_title(
        f"Fan Chart — {banco}  |  h = 1 … 90 días hábiles\n"
        f"Bandas: mediana de predicciones del modelo sobre todas las fechas del TEST  "
        f"({pd.Timestamp(fechas_test[0]).strftime('%d %b %Y')} – "
        f"{pd.Timestamp(fechas_test[-1]).strftime('%d %b %Y')})\n"
        f"Líneas: ruta realizada D−R por cada fecha t",
        fontweight="bold", fontsize=11
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(hs.min() - 1, hs.max() + 1)
    plt.tight_layout()

    nombre = f"fanchart_{banco}_h1_h90_test.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGuardado: {DIR_OUTPUT / nombre}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modelos, cols_feat = cargar_modelos(BANCO, DIR_MODELOS)
    df_train, df_test  = leer_y_split(BANCO)

    # Alinear cols_feat con las disponibles en el parquet
    cols_disponibles = [c for c in cols_feat if c in df_test.columns]
    if len(cols_disponibles) < len(cols_feat):
        faltantes = set(cols_feat) - set(cols_disponibles)
        print(f"  Advertencia: {len(faltantes)} features del modelo no están en el parquet: {faltantes}")

    preds = predecir(modelos, df_test, cols_disponibles)
    graficar(preds, BANCO)
