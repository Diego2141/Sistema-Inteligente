# -*- coding: utf-8 -*-
"""
aux_comparar_features.py
Comparativa sistemática de importancia de features para los 8 modelos.

Fuentes leídas por carpeta de modelo:
  · wfcv_v3_importancias_{banco}.csv   — gain XGBoost por fold (disponible siempre)
  · diag_gain_train_{tag}.csv          — gain promediado por fold  (si existe)
  · diag_perm_val_{tag}.csv            — block-perm OOS por fold   (si existe)
  · diag_shap_val_{tag}.csv            — SHAP OOS por fold         (si existe)

Outputs → DIR_WFCV / "comparacion_features/"
  00_tabla_consensus.csv           — rank medio por feature × modelo
  01_heatmap_consensus.png         — heatmap rank normalizado (features × modelos)
  02_top20_por_modelo.png          — top-20 gain por modelo (subplots)
  03_rank_stability.png            — mean_rank vs std_rank por modelo
  04_overfitting_gain_perm.png     — (gain − perm) por feature × modelo  [si diag OK]
  05_gain_perm_shap_consensus.png  — barras 3 señales promediadas [si diag OK]
"""

import sys
from pathlib import Path
sys.path.insert(0, r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────
BASE     = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV = BASE / "2. Output" / "step005_wfcv_v3"
DIR_OUT  = DIR_WFCV / "comparacion_features"
DIR_OUT.mkdir(parents=True, exist_ok=True)

BANCO  = "SISTEMA"
TOP_N  = 25    # features mostradas en heatmap y barras

# ─────────────────────────────────────────────────────────────────
# Helper: parsear nombre de carpeta  (igual que aux_comparar_modelos)
# ─────────────────────────────────────────────────────────────────
def _parsear(nombre: str) -> dict:
    partes = nombre.split("_")
    if partes[1] == "qt":
        modelo, resto = "xgb_qt", partes[2:]
    else:
        modelo, resto = "xgb", partes[1:]
    modo     = resto[0]
    ventanas = resto[1] if len(resto) > 1 else ""
    try:
        pv  = ventanas.split(".")
        val = float(pv[0][-1] + "." + pv[1][0])
    except Exception:
        val = float("nan")
    return {"nombre": nombre, "modelo": modelo, "modo": modo, "val_yr": val,
            "tag_diag": f"{BANCO}_{modelo}_{'exp' if modo == 'expanding' else 'rol'}"}


# ─────────────────────────────────────────────────────────────────
# Helper: leer CSV de importancias → long (feature, fold, gain_raw)
# ─────────────────────────────────────────────────────────────────
def _leer_importancias(path: Path) -> pd.DataFrame:
    """
    Formato ancho: primera columna = 'feature', resto = fold IDs numéricos.
    Devuelve long: columns [feature, fold, gain_raw].
    """
    df = pd.read_csv(path)
    feat_col = df.columns[0]              # 'feature'
    fold_cols = [c for c in df.columns[1:] if str(c).strip() != ""]
    long = df.melt(id_vars=[feat_col], value_vars=fold_cols,
                   var_name="fold", value_name="gain_raw")
    long.rename(columns={feat_col: "feature"}, inplace=True)
    long["fold"] = pd.to_numeric(long["fold"], errors="coerce")
    long.dropna(subset=["fold"], inplace=True)
    return long


# ─────────────────────────────────────────────────────────────────
# Helper: leer CSV de diagnóstico (gain/perm/shap) → long
# Acepta tanto folds-en-filas como features-en-filas.
# ─────────────────────────────────────────────────────────────────
def _leer_diag(path: Path, senal: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    first_col = df.columns[0]
    rest_cols  = df.columns[1:].tolist()

    # Detectar orientación: ¿primera columna es feature (str) o fold (int)?
    first_vals = pd.to_numeric(df[first_col], errors="coerce")
    es_fold_index = first_vals.notna().all()

    if es_fold_index:
        # Filas = folds, columnas = features
        df.rename(columns={first_col: "fold"}, inplace=True)
        long = df.melt(id_vars=["fold"], var_name="feature", value_name=senal)
    else:
        # Filas = features, columnas = folds
        df.rename(columns={first_col: "feature"}, inplace=True)
        long = df.melt(id_vars=["feature"], var_name="fold", value_name=senal)
        long["fold"] = pd.to_numeric(long["fold"], errors="coerce")

    long.dropna(subset=["fold"], inplace=True)
    return long


# ─────────────────────────────────────────────────────────────────
# Cargar todos los modelos
# ─────────────────────────────────────────────────────────────────
carpetas = sorted(
    [d for d in DIR_WFCV.iterdir() if d.is_dir() and d.name.startswith("xgb")],
    key=lambda d: d.name,
)
print(f"Carpetas encontradas: {len(carpetas)}")

imp_long  = []    # gain (wfcv_v3_importancias) — siempre disponible
diag_long = []    # gain/perm/shap diagnóstico  — cuando exista

for d in carpetas:
    meta = _parsear(d.name)

    # ── importancias (requerido) ──────────────────────────────────
    f_imp = list(d.glob(f"wfcv_v3_importancias_{BANCO}*.csv"))
    if not f_imp:
        print(f"  ⚠ Sin importancias en {d.name}")
        continue
    df_imp = _leer_importancias(f_imp[-1])
    df_imp["nombre"] = meta["nombre"]
    df_imp["modelo"] = meta["modelo"]
    df_imp["modo"]   = meta["modo"]
    df_imp["val_yr"] = meta["val_yr"]
    imp_long.append(df_imp)

    # ── diagnóstico gain/perm/shap (opcional) ────────────────────
    tag = meta["tag_diag"]
    for senal, patron in [
        ("gain_diag", f"diag_gain_train_{tag}.csv"),
        ("perm",      f"diag_perm_val_{tag}.csv"),
        ("shap",      f"diag_shap_val_{tag}.csv"),
    ]:
        f_diag = d / patron
        if not f_diag.exists():
            continue
        df_d = _leer_diag(f_diag, senal)
        if df_d.empty:
            continue
        df_d["nombre"] = meta["nombre"]
        df_d["modelo"] = meta["modelo"]
        df_d["modo"]   = meta["modo"]
        df_d["val_yr"] = meta["val_yr"]
        diag_long.append(df_d)

df_imp_all  = pd.concat(imp_long,  ignore_index=True) if imp_long  else pd.DataFrame()
df_diag_all = pd.concat(diag_long, ignore_index=True) if diag_long else pd.DataFrame()

n_diag_modelos = df_diag_all["nombre"].nunique() if not df_diag_all.empty else 0
print(f"Importancias: {len(df_imp_all)} filas — {df_imp_all['nombre'].nunique()} modelos")
print(f"Diagnóstico : {len(df_diag_all)} filas — {n_diag_modelos} modelos")

# ─────────────────────────────────────────────────────────────────
# Paleta y etiquetas cortas
# ─────────────────────────────────────────────────────────────────
nombres_ord = sorted(df_imp_all["nombre"].unique())
_pal   = {n: cm.tab10(i % 10) for i, n in enumerate(nombres_ord)}

def _etiq(nombre):
    m = _parsear(nombre)
    return (f"{'QT' if 'qt' in m['modelo'] else 'XGB'} "
            f"{'EXP' if m['modo']=='expanding' else 'ROL'} "
            f"v={m['val_yr']}y")

# ─────────────────────────────────────────────────────────────────
# Construir tabla resumen: mean_gain_raw normalizado por modelo
# ─────────────────────────────────────────────────────────────────
# Para cada (modelo, feature): media de gain_raw sobre folds
pivot_gain = (
    df_imp_all.groupby(["nombre", "feature"])["gain_raw"]
    .mean()
    .unstack("nombre")
    .fillna(0.0)
)

# Normalizar columna a columna (rank pct: 1 = top, 0 = bottom)
pivot_rank = pivot_gain.rank(ascending=True, pct=True)

# Media de rank pct sobre todos los modelos → consensus
pivot_rank["consensus"] = pivot_rank.mean(axis=1)
top_consensus = pivot_rank.sort_values("consensus", ascending=False).head(TOP_N).index.tolist()

print(f"\nTop {TOP_N} features por consensus (gain rank promedio):")
for i, f in enumerate(top_consensus, 1):
    print(f"  {i:2d}. {f:<35}  consensus={pivot_rank.loc[f,'consensus']:.3f}")

# Guardar tabla
df_tabla = pivot_rank.reset_index().rename(columns={"index": "feature"})
df_tabla.to_csv(DIR_OUT / "00_tabla_consensus.csv", index=False)

# ─────────────────────────────────────────────────────────────────
# 1. HEATMAP CONSENSUS (features × modelos, valor = rank pct)
# ─────────────────────────────────────────────────────────────────
_cmap_feat = mcolors.LinearSegmentedColormap.from_list(
    "feat", ["#f7f7f7", "#fee090", "#fc8d59", "#d73027"], N=256)

mat = pivot_rank.loc[top_consensus, nombres_ord].values

fig1, ax1 = plt.subplots(figsize=(max(10, len(nombres_ord) * 1.4), TOP_N * 0.42 + 2))
im = ax1.imshow(mat, aspect="auto", cmap=_cmap_feat,
                vmin=0, vmax=1, origin="upper")
ax1.set_xticks(np.arange(len(nombres_ord)))
ax1.set_xticklabels([_etiq(n) for n in nombres_ord], rotation=30, ha="right", fontsize=8)
ax1.set_yticks(np.arange(len(top_consensus)))
ax1.set_yticklabels(top_consensus, fontsize=8)
ax1.set_title(
    f"Importancia Relativa por Feature y Modelo — {BANCO}\n"
    f"(rank percentil del gain medio sobre folds; 1.0 = más importante en ese modelo)",
    fontweight="bold", fontsize=10)
plt.colorbar(im, ax=ax1, shrink=0.6,
             format=mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

# Anotar valores en celdas del top 10
for r, feat in enumerate(top_consensus[:10]):
    for c, nom in enumerate(nombres_ord):
        v = pivot_rank.loc[feat, nom]
        ax1.text(c, r, f"{v:.2f}", ha="center", va="center",
                 fontsize=5.5, color="black" if v < 0.8 else "white")

plt.tight_layout()
plt.savefig(DIR_OUT / "01_heatmap_consensus.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 2. TOP 20 GAIN POR MODELO (subplots 4×2)
# ─────────────────────────────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(len(nombres_ord) / ncols))
fig2, axes2 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 5, nrows * 4.5),
                            sharey=False)
axes2_flat = axes2.flatten()

for idx, nombre in enumerate(nombres_ord):
    ax = axes2_flat[idx]
    sub = (df_imp_all[df_imp_all["nombre"] == nombre]
           .groupby("feature")["gain_raw"].mean()
           .sort_values(ascending=False)
           .head(20))
    # Normalizar para visualizar
    sub_norm = sub / sub.max() if sub.max() > 0 else sub
    colors   = [_pal[nombre]] * len(sub_norm)
    ax.barh(range(len(sub_norm)), sub_norm.values[::-1],
            color=_pal[nombre], alpha=0.80, edgecolor="white")
    ax.set_yticks(range(len(sub_norm)))
    ax.set_yticklabels(sub_norm.index[::-1], fontsize=7)
    ax.set_title(_etiq(nombre), fontsize=8, fontweight="bold")
    ax.set_xlabel("Gain normalizado", fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)

for j in range(idx + 1, len(axes2_flat)):
    axes2_flat[j].set_visible(False)

plt.suptitle(f"Top 20 Features por Modelo — Gain XGBoost (TRAIN) — {BANCO}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_OUT / "02_top20_por_modelo.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 3. RANK STABILITY (mean_rank vs std_rank por feature y modelo)
# ─────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4.5, nrows * 4),
                            sharey=False)
axes3_flat = axes3.flatten()

for idx, nombre in enumerate(nombres_ord):
    ax = axes3_flat[idx]
    sub = df_imp_all[df_imp_all["nombre"] == nombre].copy()

    # Rango dentro de cada fold (1 = más importante)
    sub["rank_fold"] = sub.groupby("fold")["gain_raw"].rank(ascending=False)
    stats = sub.groupby("feature")["rank_fold"].agg(["mean", "std"]).fillna(0)
    stats = stats[stats["mean"] <= 30]   # solo features que entran al top-30 en promedio

    sc = ax.scatter(stats["mean"], stats["std"],
                    s=40, alpha=0.7, color=_pal[nombre], edgecolors="black", lw=0.4)

    # Etiquetar las top-10 más importantes
    top10 = stats.sort_values("mean").head(10)
    for feat, row in top10.iterrows():
        ax.annotate(feat, (row["mean"], row["std"]),
                    fontsize=5.5, textcoords="offset points",
                    xytext=(4, 2), color="black")

    ax.set_xlabel("Rank medio (1 = más importante)", fontsize=7)
    ax.set_ylabel("Std del rank entre folds", fontsize=7)
    ax.set_title(_etiq(nombre), fontsize=8, fontweight="bold")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.annotate("← más importante", xy=(0.98, 0.03),
                xycoords="axes fraction", ha="right", fontsize=6, color="gray")

for j in range(idx + 1, len(axes3_flat)):
    axes3_flat[j].set_visible(False)

plt.suptitle(f"Estabilidad del Ranking por Fold — {BANCO}\n"
             f"Esquina inferior-derecha = consistentemente importante",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_OUT / "03_rank_stability.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────
# 4 & 5. DIAGNÓSTICO GAIN/PERM/SHAP (solo si los CSVs existen)
# ─────────────────────────────────────────────────────────────────
if df_diag_all.empty:
    print("\n  [diag] Sin archivos diag_gain/perm/shap → plots 4 y 5 omitidos.")
else:
    senales_disp = [s for s in ["gain_diag", "perm", "shap"]
                    if s in df_diag_all.columns]
    print(f"\n  [diag] Señales disponibles: {senales_disp} "
          f"— modelos: {df_diag_all['nombre'].nunique()}")

    # Media por (nombre, feature) para cada señal
    diag_mean = (
        df_diag_all.groupby(["nombre", "feature"])[senales_disp]
        .mean()
        .reset_index()
    )

    # Normalizar señales 0-1 dentro de cada modelo
    for senal in senales_disp:
        diag_mean[f"{senal}_norm"] = (
            diag_mean.groupby("nombre")[senal]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))
        )

    # ── Plot 4: (gain − perm) por feature × modelo ──────────────
    if "gain_diag" in senales_disp and "perm" in senales_disp:
        diag_mean["overfit_ratio"] = (
            diag_mean["gain_diag_norm"] - diag_mean["perm_norm"]
        )

        # Top features por overfit promedio entre modelos
        top_overfit = (
            diag_mean.groupby("feature")["overfit_ratio"]
            .mean()
            .sort_values(ascending=False)
            .head(TOP_N)
        )

        mod_diag = sorted(diag_mean["nombre"].unique())
        mat_of = (
            diag_mean[diag_mean["feature"].isin(top_overfit.index)]
            .pivot_table(index="feature", columns="nombre",
                         values="overfit_ratio", aggfunc="mean")
            .reindex(top_overfit.index)
            .fillna(0)
        )

        _cmap_div = mcolors.LinearSegmentedColormap.from_list(
            "div", ["#2166ac", "#f7f7f7", "#d73027"], N=256)

        fig4, ax4 = plt.subplots(
            figsize=(max(9, len(mod_diag) * 1.5), TOP_N * 0.42 + 2))
        _vlim = max(abs(mat_of.values.max()), abs(mat_of.values.min()), 0.3)
        im4 = ax4.imshow(mat_of.values, aspect="auto", cmap=_cmap_div,
                         vmin=-_vlim, vmax=_vlim, origin="upper")
        ax4.set_xticks(np.arange(len(mod_diag)))
        ax4.set_xticklabels([_etiq(n) for n in mod_diag],
                            rotation=30, ha="right", fontsize=8)
        ax4.set_yticks(np.arange(len(mat_of)))
        ax4.set_yticklabels(mat_of.index, fontsize=8)
        ax4.set_title(
            f"Señal de Overfitting: gain_norm − perm_norm — {BANCO}\n"
            "Rojo = gain alto pero perm bajo (memorización); azul = perm > gain (OOS útil)",
            fontweight="bold", fontsize=10)
        plt.colorbar(im4, ax=ax4, shrink=0.6)
        plt.tight_layout()
        plt.savefig(DIR_OUT / "04_overfitting_gain_perm.png", dpi=150,
                    bbox_inches="tight")
        plt.show()

    # ── Plot 5: barras gain/perm/shap consenso (top features) ───
    if len(senales_disp) >= 2:
        top_feats_diag = (
            diag_mean.groupby("feature")["gain_diag_norm"]
            .mean().sort_values(ascending=False)
            .head(TOP_N).index.tolist()
            if "gain_diag_norm" in diag_mean.columns
            else top_consensus[:TOP_N]
        )

        # Promedio sobre todos los modelos disponibles para cada señal
        diag_consensus = (
            diag_mean[diag_mean["feature"].isin(top_feats_diag)]
            .groupby("feature")[[f"{s}_norm" for s in senales_disp]]
            .mean()
            .reindex(top_feats_diag)
        )

        fig5, ax5 = plt.subplots(figsize=(12, TOP_N * 0.45 + 2))
        y      = np.arange(len(diag_consensus))
        height = 0.25
        colores_senales = {"gain_diag_norm": "steelblue",
                           "perm_norm":      "seagreen",
                           "shap_norm":      "tomato"}
        etiq_senales    = {"gain_diag_norm": "Gain (TRAIN)",
                           "perm_norm":      "Block-perm (VAL OOS)",
                           "shap_norm":      "SHAP |mean| (VAL OOS)"}
        offsets = np.linspace(-height, height, len(senales_disp))

        for i, senal in enumerate([f"{s}_norm" for s in senales_disp]):
            if senal not in diag_consensus.columns:
                continue
            vals = diag_consensus[senal].values
            ax5.barh(y + offsets[i], vals,
                     height=height * 0.85,
                     color=colores_senales.get(senal, "gray"),
                     alpha=0.80, label=etiq_senales.get(senal, senal))

        ax5.set_yticks(y)
        ax5.set_yticklabels(diag_consensus.index, fontsize=8)
        ax5.set_xlabel("Importancia normalizada (promedio sobre todos los modelos)")
        ax5.set_title(
            f"Gain / Perm / SHAP — Consensus sobre {n_diag_modelos} modelos — {BANCO}\n"
            "convergencia gain≈perm≈SHAP → feature genuinamente útil en todos los modelos",
            fontweight="bold", fontsize=10)
        ax5.legend(fontsize=8, loc="lower right")
        ax5.invert_yaxis()
        ax5.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(DIR_OUT / "05_gain_perm_shap_consensus.png", dpi=150,
                    bbox_inches="tight")
        plt.show()

print(f"\n  Outputs en: {DIR_OUT}")
