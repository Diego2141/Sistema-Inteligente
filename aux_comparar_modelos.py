# -*- coding: utf-8 -*-
"""
aux_comparar_modelos.py
Comparativa sistemática de los 8 modelos walk-forward CV generados por
step005_iteracion.py.

Lee los CSVs de cada carpeta (metricas TEST/VAL y por_h TEST/VAL) y produce:
  1. Tabla resumen: coverage_90, pinball_q50, winkler_90, crps, gap VAL-TEST
  2. Boxplots por fold: cobertura y pinball para todos los modelos
  3. Coverage por buckets de horizonte (h=1-25 / 26-50 / 51-75)
  4. Heatmap coverage(fold × h) por modelo
  5. Mapa de calor comparativo VAL vs TEST
  6. Ranking final con recomendación

Output → DIR_WFCV / "comparacion_modelos/"
"""

import sys
from pathlib import Path

sys.path.insert(0, r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────
BASE     = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV = BASE / "2. Output" / "step005_wfcv_v3"
DIR_OUT  = DIR_WFCV / "comparacion_modelos"
DIR_OUT.mkdir(parents=True, exist_ok=True)

BANCO = "SISTEMA"   # banco objetivo
TARGET_COV = 0.90   # cobertura nominal

# ─────────────────────────────────────────────────────────────────
# Helper: parsear nombre de carpeta
# ─────────────────────────────────────────────────────────────────
def _parsear_nombre(nombre: str) -> dict:
    """
    Ejemplos:
      xgb_expanding_50.51  → modelo=xgb,  modo=expanding, val=0.5
      xgb_qt_rolling_51.01 → modelo=xgb_qt, modo=rolling, val=1.0
    """
    partes = nombre.split("_")
    if partes[1] == "qt":
        modelo = "xgb_qt"
        resto  = partes[2:]
    else:
        modelo = "xgb"
        resto  = partes[1:]
    modo   = resto[0]                       # expanding / rolling
    ventanas = resto[1] if len(resto) > 1 else ""
    # ventanas = "50.51" → train=5, val=0.5, test=1
    # ventanas = "51.01" → train=5, val=1.0, test=1
    try:
        # "50.51" → before="50", after="51" → val = float("0.5") = 0.5
        # "51.01" → before="51", after="01" → val = float("1.0") = 1.0
        partes_v = ventanas.split(".")
        val = float(partes_v[0][-1] + "." + partes_v[1][0]) if len(partes_v) > 1 else float("nan")
    except Exception:
        val = float("nan")
    return {"modelo": modelo, "modo": modo, "val_yr": val, "nombre": nombre}


# ─────────────────────────────────────────────────────────────────
# Cargar CSVs de todos los modelos
# ─────────────────────────────────────────────────────────────────
carpetas = sorted(
    [d for d in DIR_WFCV.iterdir()
     if d.is_dir() and d.name.startswith("xgb")],
    key=lambda d: d.name,
)

print(f"Carpetas encontradas: {len(carpetas)}")
for c in carpetas:
    print(f"  {c.name}")

registros    = []
por_h_test   = {}
por_h_val    = {}

for d in carpetas:
    meta = _parsear_nombre(d.name)

    f_met_test = sorted(d.glob(f"wfcv_v3_metricas_TEST_{BANCO}_*.csv"))
    f_met_val  = sorted(d.glob(f"wfcv_v3_metricas_VAL_{BANCO}_*.csv"))
    f_ph_test  = sorted(d.glob(f"wfcv_v3_por_h_TEST_{BANCO}_*.csv"))
    f_ph_val   = sorted(d.glob(f"wfcv_v3_por_h_VAL_{BANCO}_*.csv"))

    if not f_met_test:
        print(f"  ⚠ Sin métricas TEST en {d.name}")
        continue

    df_t = pd.read_csv(f_met_test[-1])
    df_v = pd.read_csv(f_met_val[-1])  if f_met_val  else pd.DataFrame()

    for df, split in [(df_t, "test"), (df_v, "val")]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            registros.append({
                "nombre"    : meta["nombre"],
                "modelo"    : meta["modelo"],
                "modo"      : meta["modo"],
                "val_yr"    : meta["val_yr"],
                "split"     : split,
                "fold"      : row.get("fold", np.nan),
                "coverage_90": row.get("coverage_90", np.nan),
                "pinball_q50": row.get("pinball_q50", np.nan),
                "winkler_90" : row.get("winkler_90",  np.nan),
                "crps_approx": row.get("crps_approx", np.nan),
            })

    if f_ph_test:
        por_h_test[meta["nombre"]] = pd.read_csv(f_ph_test[-1])
    if f_ph_val:
        por_h_val[meta["nombre"]]  = pd.read_csv(f_ph_val[-1])

df_all = pd.DataFrame(registros)
print(f"\nRegistros cargados: {len(df_all)}  |  modelos: {df_all['nombre'].nunique()}")

# ─────────────────────────────────────────────────────────────────
# Paleta de colores por modelo
# ─────────────────────────────────────────────────────────────────
nombres_ord = sorted(df_all["nombre"].unique())
import matplotlib.cm as cm
_pal = {n: cm.tab10(i % 10) for i, n in enumerate(nombres_ord)}

# Etiquetas cortas
_etiq = {}
for n in nombres_ord:
    m = _parsear_nombre(n)
    _etiq[n] = f"{'QT' if 'qt' in m['modelo'] else 'XGB'}\n{m['modo'][:3].upper()}\nval={m['val_yr']}y"

# ─────────────────────────────────────────────────────────────────
# 1. TABLA RESUMEN
# ─────────────────────────────────────────────────────────────────
resumen_rows = []
for nombre in nombres_ord:
    dt = df_all[(df_all["nombre"] == nombre) & (df_all["split"] == "test")]
    dv = df_all[(df_all["nombre"] == nombre) & (df_all["split"] == "val")]

    cov_t  = dt["coverage_90"].mean()
    pin_t  = dt["pinball_q50"].mean()
    wink_t = dt["winkler_90"].mean()
    crps_t = dt["crps_approx"].mean()
    std_t  = dt["coverage_90"].std()

    cov_v  = dv["coverage_90"].mean() if not dv.empty else np.nan
    gap    = cov_v - cov_t if not np.isnan(cov_v) else np.nan
    desvio = abs(cov_t - TARGET_COV)

    resumen_rows.append({
        "Modelo"         : nombre,
        "Tipo"           : _parsear_nombre(nombre)["modelo"].upper(),
        "Modo"           : _parsear_nombre(nombre)["modo"].capitalize(),
        "Val (yr)"       : _parsear_nombre(nombre)["val_yr"],
        "Cov TEST"       : round(cov_t,  4),
        "Cov VAL"        : round(cov_v,  4),
        "Gap VAL-TEST"   : round(gap,    4),
        "|Cov-90%|"      : round(desvio, 4),
        "Std Cov (folds)": round(std_t,  4),
        "Pinball Q50"    : round(pin_t,  0),
        "Winkler 90"     : round(wink_t, 0),
        "CRPS"           : round(crps_t, 0),
    })

df_res = pd.DataFrame(resumen_rows).sort_values("|Cov-90%|")

print("\n" + "═" * 100)
print("TABLA RESUMEN — ordenada por |Cov TEST − 90%|  (menor = mejor calibrado)")
print("═" * 100)
print(df_res.to_string(index=False))

ruta_res = DIR_OUT / "00_tabla_resumen.csv"
df_res.to_csv(ruta_res, index=False)
print(f"\n  Guardado: {ruta_res}")

# ─────────────────────────────────────────────────────────────────
# 2. BOXPLOT: Coverage 90% por fold — todos los modelos
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, split, titulo in [
    (axes[0], "test", "Coverage 90% — TEST OOS"),
    (axes[1], "val",  "Coverage 90% — VAL (Optuna)"),
]:
    data_box = []
    labels   = []
    colors   = []
    for nombre in nombres_ord:
        vals = df_all[(df_all["nombre"] == nombre) & (df_all["split"] == split)]["coverage_90"].dropna()
        if vals.empty:
            continue
        data_box.append(vals.values)
        labels.append(_etiq[nombre])
        colors.append(_pal[nombre])

    bp = ax.boxplot(data_box, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(TARGET_COV, color="red", lw=1.5, ls="--", label="90% nominal")
    ax.axhspan(TARGET_COV - 0.02, TARGET_COV + 0.02, color="red", alpha=0.07)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("Coverage 90%")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle(f"Calibración por Fold — {BANCO}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_OUT / "01_boxplot_coverage.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 3. BOXPLOT: Pinball Q50 por fold
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, split, titulo in [
    (axes[0], "test", "Pinball Q50 — TEST OOS"),
    (axes[1], "val",  "Pinball Q50 — VAL (Optuna)"),
]:
    data_box = []
    labels   = []
    colors   = []
    for nombre in nombres_ord:
        vals = df_all[(df_all["nombre"] == nombre) & (df_all["split"] == split)]["pinball_q50"].dropna()
        if vals.empty:
            continue
        data_box.append(vals.values)
        labels.append(_etiq[nombre])
        colors.append(_pal[nombre])

    bp = ax.boxplot(data_box, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("Pinball Q50 (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle(f"Error de Pronóstico Puntual — {BANCO}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_OUT / "02_boxplot_pinball.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 4. COVERAGE POR BUCKETS DE HORIZONTE
# ─────────────────────────────────────────────────────────────────
BUCKETS = [(1, 25, "Corto  (h=1–25)"),
           (26, 50, "Medio  (h=26–50)"),
           (51, 75, "Largo  (h=51–75)")]

if por_h_test:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (h_lo, h_hi, label) in zip(axes, BUCKETS):
        vals_m = []
        vals_e = []
        labels = []
        colors = []
        for nombre in nombres_ord:
            if nombre not in por_h_test:
                continue
            df_ph = por_h_test[nombre]
            mask  = (df_ph["h"] >= h_lo) & (df_ph["h"] <= h_hi)
            sub   = df_ph[mask].groupby("fold")["coverage_90"].mean()
            if sub.empty:
                continue
            vals_m.append(sub.mean())
            vals_e.append(sub.std() / np.sqrt(len(sub)))
            labels.append(_etiq[nombre])
            colors.append(_pal[nombre])

        x = np.arange(len(vals_m))
        bars = ax.bar(x, vals_m, yerr=vals_e, capsize=4,
                      color=colors, alpha=0.8, edgecolor="white", width=0.65,
                      error_kw=dict(ecolor="black", lw=1.2))
        ax.axhline(TARGET_COV, color="red", lw=1.5, ls="--", alpha=0.8)
        ax.axhspan(TARGET_COV - 0.02, TARGET_COV + 0.02, color="red", alpha=0.07)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0.5, 1.05)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Coverage 90% (media ± 1SE por fold)")

    plt.suptitle(f"Coverage 90% por Horizonte de Predicción — {BANCO}  [TEST OOS]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIR_OUT / "03_coverage_por_horizonte.png", dpi=150, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────────────────────────
# 5. HEATMAP coverage(fold × h) — uno por modelo
# ─────────────────────────────────────────────────────────────────
if por_h_test:
    n_mod = len([n for n in nombres_ord if n in por_h_test])
    ncols = 4
    nrows = int(np.ceil(n_mod / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes_flat = axes.flatten() if n_mod > 1 else [axes]

    _cmap = mcolors.LinearSegmentedColormap.from_list(
        "cov", ["#d73027", "#fee090", "#91cf60", "#1a9850"], N=256)
    _norm = mcolors.Normalize(vmin=0.60, vmax=1.0)

    plot_idx = 0
    for nombre in nombres_ord:
        if nombre not in por_h_test:
            continue
        ax  = axes_flat[plot_idx]
        df_ph = por_h_test[nombre]
        pivot = (df_ph.groupby(["fold", "h"])["coverage_90"]
                 .mean()
                 .unstack("h")
                 .sort_index())
        im = ax.imshow(pivot.values, aspect="auto", cmap=_cmap, norm=_norm,
                       origin="lower")
        ax.set_xticks(np.arange(0, pivot.shape[1], 10))
        ax.set_xticklabels(pivot.columns[::10], fontsize=6, rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"F{f}" for f in pivot.index], fontsize=6)
        ax.set_title(nombre.replace("_", " "), fontsize=8, fontweight="bold")
        ax.set_xlabel("h (días hábiles)", fontsize=7)
        ax.set_ylabel("Fold", fontsize=7)
        ax.axhline(len(pivot.index) - 0.5, color="none")
        plt.colorbar(im, ax=ax,
                     format=mticker.FuncFormatter(lambda x, _: f"{x:.0%}"),
                     shrink=0.8)
        plot_idx += 1

    for j in range(plot_idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(f"Heatmap Coverage 90%  (fold × horizonte) — TEST OOS — {BANCO}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIR_OUT / "04_heatmap_coverage_fold_h.png", dpi=150, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────────────────────────
# 6. EVOLUCIÓN TEMPORAL: Coverage TEST por fold (líneas)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, split, titulo in [
    (axes[0], "test", "Coverage 90% por Fold — TEST OOS"),
    (axes[1], "val",  "Coverage 90% por Fold — VAL"),
]:
    for nombre in nombres_ord:
        sub = df_all[(df_all["nombre"] == nombre) & (df_all["split"] == split)]
        if sub.empty:
            continue
        sub = sub.sort_values("fold")
        ax.plot(sub["fold"], sub["coverage_90"], "o-",
                color=_pal[nombre], lw=1.5, ms=5,
                label=nombre.replace("_", " "))

    ax.axhline(TARGET_COV, color="red", lw=1.5, ls="--", alpha=0.7, label="90% nominal")
    ax.axhspan(TARGET_COV - 0.02, TARGET_COV + 0.02, color="red", alpha=0.07)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Coverage 90%")
    ax.set_title(titulo, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=6.5, loc="lower left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.suptitle(f"Estabilidad de Calibración por Fold — {BANCO}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_OUT / "05_coverage_por_fold_linea.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 7. GRÁFICO SCATTER: Coverage TEST vs Pinball Q50 (pareto)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for i, nombre in enumerate(nombres_ord):
    dt = df_all[(df_all["nombre"] == nombre) & (df_all["split"] == "test")]
    cov = dt["coverage_90"].mean()
    pin = dt["pinball_q50"].mean()
    if np.isnan(cov) or np.isnan(pin):
        continue
    ax.scatter(pin, cov, s=200, color=_pal[nombre], zorder=4, edgecolors="black", lw=0.8)
    ax.annotate(nombre.replace("xgb_qt", "QT").replace("xgb", "XGB"),
                (pin, cov), textcoords="offset points", xytext=(8, 4), fontsize=7)

ax.axhline(TARGET_COV,    color="red",    lw=1.2, ls="--", alpha=0.7, label="Cov 90% nominal")
ax.axhspan(TARGET_COV - 0.02, TARGET_COV + 0.02, color="red", alpha=0.06)
ax.set_xlabel("Pinball Q50 — TEST (USD, menor = mejor)")
ax.set_ylabel("Coverage 90% — TEST")
ax.set_title(f"Trade-off Calibración vs Error Puntual — {BANCO}\n"
             f"Óptimo: cerca de la línea roja y lo más a la izquierda posible",
             fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(DIR_OUT / "06_scatter_calibracion_vs_pinball.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 8. RANKING FINAL
# ─────────────────────────────────────────────────────────────────
print("\n" + "═" * 90)
print("RANKING FINAL — criterios ponderados")
print("  1. |Cov_TEST − 90%|  → peso 40%  (calibración, principal)")
print("  2. Pinball Q50 TEST  → peso 30%  (precisión puntual, normalizado)")
print("  3. Std Cov (folds)   → peso 20%  (estabilidad temporal)")
print("  4. |Gap VAL-TEST|    → peso 10%  (sobreajuste a validación)")
print("═" * 90)

df_rank = df_res.copy()

# Normalizar cada criterio a [0, 1] donde 0 = mejor
def _norm01(s):
    r = s - s.min()
    return r / r.max() if r.max() > 0 else r

df_rank["score_cov"]  = _norm01(df_rank["|Cov-90%|"])
df_rank["score_pin"]  = _norm01(df_rank["Pinball Q50"])
df_rank["score_std"]  = _norm01(df_rank["Std Cov (folds)"])
df_rank["score_gap"]  = _norm01(df_rank["Gap VAL-TEST"].abs())

df_rank["Score Final"] = (
    0.40 * df_rank["score_cov"] +
    0.30 * df_rank["score_pin"] +
    0.20 * df_rank["score_std"] +
    0.10 * df_rank["score_gap"]
)
df_rank = df_rank.sort_values("Score Final")

cols_show = ["Modelo", "Cov TEST", "Gap VAL-TEST", "|Cov-90%|",
             "Std Cov (folds)", "Pinball Q50", "Score Final"]
print(df_rank[cols_show].to_string(index=False))

ruta_rank = DIR_OUT / "07_ranking_final.csv"
df_rank[cols_show].to_csv(ruta_rank, index=False)

# Ganador
ganador = df_rank.iloc[0]["Modelo"]
print(f"\n  ★  RECOMENDACIÓN: {ganador}")
print(f"     Score final (menor = mejor): {df_rank.iloc[0]['Score Final']:.3f}")
print(f"     Coverage TEST: {df_rank.iloc[0]['Cov TEST']:.1%}  |  "
      f"Pinball Q50: {df_rank.iloc[0]['Pinball Q50']:,.0f}")
print(f"\n  Outputs guardados en: {DIR_OUT}")
