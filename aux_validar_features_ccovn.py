# -*- coding: utf-8 -*-
"""
aux_validar_features_ccovn.py
Validación estadística de los features derivados de la identidad Δsaldo≈flujo.

Pruebas (solo numpy + pandas, sin scipy ni sklearn):
  1. Correlación Pearson y Spearman por horizonte h (señal vs ruido)
  2. OLS univariado:  target ~ feature + h  → coef, t-stat, R², ΔR²
  3. Correlación rolling 252d en el tiempo  → estabilidad de la señal

Outputs → DIR_OUTPUT /
    00_corr_por_h.png         — Pearson + Spearman por bucket de h
    01_corr_lineal_h.png      — correlación por cada h individual (2..75)
    02_rolling_corr.png       — rolling 252d en el tiempo
    resultados_ccovn.xlsx     — tablas completas (3 hojas)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

###############################################################################
# CONFIGURACIÓN
###############################################################################

BASE         = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_PARQUET = BASE / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE / "2. Output" / "aux_validar_features_ccovn"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO_REF = "SISTEMA"   # banco sobre el que se calcula (usa datos sistémicos)

# Cada entrada es una lista de ALTERNATIVAS para el mismo concepto: primero el
# nombre de v1, después el de v2. step001 v2 renombró las columnas CCOVN de
# naming absoluto (ccovn_bbva_lag1) a relativo (ccovn_propio_lag1 /
# ccovn_contraparte_lag1), así que los nombres de v1 no existen en una matriz
# generada con v2. Se toma la primera alternativa que esté presente Y no sea
# todo-NaN; si ninguna aparece, se avisa en vez de omitirla en silencio.
#
# Sobre la entidad SISTEMA, el equivalente de bbva_share_lag1 es
# share_contraparte_lag1 (el FOCO sobre el total), NO share_propio_lag1: v2 anula
# ese a NaN a propósito cuando la entidad es el sistema, porque ahí "propio" y
# "sistema" son la misma serie y el share daría 1 constante.

# Features a validar — se omiten automáticamente si no están en la matriz
FEATURES_NUEVOS = [
    ["var_ccovn_bbva_exceso_lag1", "var_ccovn_propio_exceso_lag1"],
    ["ccovn_vs_dia_mes_lag1"],
    ["residuo_ccovn_lag1"],
]

# Features existentes de ccovn usados como base de comparación en ΔR².
# Que falte uno acá NO es cosmético: empobrece el baseline y por lo tanto
# INFLA delta_r2_vs_base de todos los features nuevos.
FEATURES_BASE_CCOVN = [
    ["ccovn_sistema_lag1"],
    ["var_ccovn_sistema_lag1"],
    ["bbva_share_lag1", "share_contraparte_lag1"],
]

H_BUCKETS = {
    "corto (h=2-5)":   (2,  5),
    "medio (h=6-20)":  (6,  20),
    "largo (h=21-75)": (21, 75),
}

ROLLING_WINDOW = 252  # días hábiles (~1 año)
MIN_OBS        = 30   # mínimo de observaciones para reportar un estadístico

###############################################################################
# FUNCIONES ESTADÍSTICAS (sin scipy)
###############################################################################

def _mask_finite(*arrays):
    """Devuelve máscara booleana donde todos los arrays son finitos."""
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def pearson(x: np.ndarray, y: np.ndarray):
    """Pearson r y t-stat. Devuelve (r, t, n) o (nan, nan, 0)."""
    m = _mask_finite(x, y)
    n = m.sum()
    if n < MIN_OBS:
        return np.nan, np.nan, n
    xc = x[m] - x[m].mean()
    yc = y[m] - y[m].mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    r = (xc @ yc) / denom if denom > 0 else np.nan
    t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-15)) if np.isfinite(r) else np.nan
    return r, t, n


def spearman(x: np.ndarray, y: np.ndarray):
    """Spearman r y t-stat. Devuelve (r, t, n) o (nan, nan, 0)."""
    m = _mask_finite(x, y)
    n = m.sum()
    if n < MIN_OBS:
        return np.nan, np.nan, n
    def _rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(n)
        return r
    rx = _rank(x[m]) - (n - 1) / 2
    ry = _rank(y[m]) - (n - 1) / 2
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    r = (rx @ ry) / denom if denom > 0 else np.nan
    t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-15)) if np.isfinite(r) else np.nan
    return r, t, n


def ols(y: np.ndarray, X: np.ndarray):
    """
    OLS. X NO incluye columna de intercept (se agrega aquí).
    Devuelve dict con beta, t_stat, r2, r2_adj, n o None si singular.
    """
    m = _mask_finite(y) & np.all(np.isfinite(X), axis=1)
    n = m.sum()
    k_orig = X.shape[1]
    if n < k_orig + 5:
        return None
    y_, X_ = y[m], np.column_stack([np.ones(n), X[m]])
    k = X_.shape[1]
    XtX = X_.T @ X_
    try:
        beta = np.linalg.solve(XtX, X_.T @ y_)
    except np.linalg.LinAlgError:
        return None
    resid  = y_ - X_ @ beta
    ss_res = resid @ resid
    ss_tot = ((y_ - y_.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)
    sigma2 = ss_res / max(n - k, 1)
    try:
        se     = np.sqrt(np.maximum(np.diag(sigma2 * np.linalg.inv(XtX)), 0))
        t_stat = beta / np.where(se > 0, se, np.nan)
    except np.linalg.LinAlgError:
        t_stat = np.full(k, np.nan)
    return {"beta": beta, "t_stat": t_stat, "r2": r2, "r2_adj": r2_adj, "n": int(n)}


def delta_r2(y: np.ndarray, X_base: np.ndarray, X_new_col: np.ndarray):
    """R² del modelo base vs modelo ampliado con una columna nueva."""
    res_base = ols(y, X_base)
    res_aug  = ols(y, np.column_stack([X_base, X_new_col]))
    if res_base is None or res_aug is None:
        return np.nan, np.nan
    return res_aug["r2"] - res_base["r2"], res_aug["r2_adj"] - res_base["r2_adj"]

###############################################################################
# CARGA DE DATOS
###############################################################################

print("=" * 65)
print("  aux_validar_features_ccovn")
print("=" * 65)

if not RUTA_PARQUET.exists():
    raise FileNotFoundError(
        f"Matriz no encontrada: {RUTA_PARQUET}\n"
        "Ejecutar step001_build_feature_matrix.py primero."
    )

print(f"\n  Cargando matriz para banco='{BANCO_REF}'...")
try:
    df_full = pd.read_parquet(
        RUTA_PARQUET,
        filters=[("banco", "==", BANCO_REF)],
    )
except Exception:
    df_full = pd.read_parquet(RUTA_PARQUET)
    df_full = df_full[df_full["banco"] == BANCO_REF]

df_full = df_full.sort_values(["fecha_t", "h"]).reset_index(drop=True)
print(f"  Filas cargadas: {len(df_full):,}  |  h=[{df_full['h'].min()}..{df_full['h'].max()}]"
      f"  |  fechas: {df_full['fecha_t'].min().date()} → {df_full['fecha_t'].max().date()}")

# Filtrar features disponibles efectivamente en la matriz
def resolver_alternativas(grupos, df):
    """Resuelve cada grupo de alternativas al primer nombre presente y utilizable.

    Utilizable = está en la matriz y tiene al menos un valor no-NaN. La segunda
    condición importa: v2 materializa en NaN las columnas que no aplican a una
    entidad (share_propio_lag1 sobre SISTEMA, por ejemplo), así que "la columna
    existe" no implica "la columna trae señal". Sin este filtro, una columna
    todo-NaN entra al baseline como si aportara y el ΔR² queda igual de inflado
    que si faltara, pero sin aviso.

    Devuelve (resueltos, no_resueltos) donde no_resueltos lista los grupos
    completos que no pudieron satisfacerse, para poder reportarlos.
    """
    resueltos, no_resueltos = [], []
    for alts in grupos:
        elegido = None
        for nombre in alts:
            if nombre in df.columns and df[nombre].notna().any():
                elegido = nombre
                break
        if elegido is not None:
            resueltos.append(elegido)
        else:
            no_resueltos.append(alts)
    return resueltos, no_resueltos


feats_disp, feats_miss = resolver_alternativas(FEATURES_NUEVOS, df_full)
base_disp,  base_miss  = resolver_alternativas(FEATURES_BASE_CCOVN, df_full)

def _fmt(alts):
    return alts[0] if len(alts) == 1 else " | ".join(alts)

if feats_miss:
    print("\n  AVISO: features no encontrados en la matriz (¿regenerar parquet?):")
    for alts in feats_miss:
        print(f"    · {_fmt(alts)}")
if base_miss:
    # Antes esto se filtraba en silencio. Un baseline incompleto no rompe nada
    # visible: solo hace que delta_r2_vs_base sobreestime el aporte de los
    # features nuevos, y el número se lee igual de creíble.
    print("\n  AVISO: features BASE no encontrados. delta_r2_vs_base se mide")
    print("  contra un baseline INCOMPLETO y por lo tanto SOBREESTIMA el aporte:")
    for alts in base_miss:
        print(f"    · {_fmt(alts)}")
if not feats_disp:
    raise SystemExit("  No hay features nuevos disponibles. Abortando.")

print(f"\n  Features a validar  : {feats_disp}")
print(f"  Features base ccovn : {base_disp}")

y_all = df_full["target"].to_numpy(dtype=float)
h_all = df_full["h"].to_numpy(dtype=float)

###############################################################################
# ANÁLISIS 1 — CORRELACIÓN POR BUCKET DE H
###############################################################################

print("\n  [1/3] Correlación por bucket de h ...")

rows_buckets = []
for feat in feats_disp:
    x_all = df_full[feat].to_numpy(dtype=float)
    for bucket_name, (h_lo, h_hi) in H_BUCKETS.items():
        mask_h = (h_all >= h_lo) & (h_all <= h_hi)
        r_p, t_p, n_p = pearson(x_all[mask_h], y_all[mask_h])
        r_s, t_s, _   = spearman(x_all[mask_h], y_all[mask_h])
        rows_buckets.append({
            "feature":  feat,
            "bucket":   bucket_name,
            "pearson_r": round(r_p, 4) if np.isfinite(r_p) else None,
            "pearson_t": round(t_p, 2) if np.isfinite(t_p) else None,
            "spearman_r":round(r_s, 4) if np.isfinite(r_s) else None,
            "spearman_t":round(t_s, 2) if np.isfinite(t_s) else None,
            "n_obs":     n_p,
        })

df_buckets = pd.DataFrame(rows_buckets)

###############################################################################
# ANÁLISIS 2 — OLS UNIVARIADO Y ΔR²
###############################################################################

print("  [2/3] OLS univariado y ΔR² ...")

# Base: solo h (modelo trivial de referencia)
X_h_only  = h_all.reshape(-1, 1)
# Base ampliada: h + features ccovn existentes
X_base_ext = np.column_stack([h_all] + [
    df_full[f].to_numpy(dtype=float) for f in base_disp
]) if base_disp else X_h_only

rows_ols = []
for feat in feats_disp:
    x_feat = df_full[feat].to_numpy(dtype=float)
    # OLS completo: target ~ intercept + h + feature
    res = ols(y_all, np.column_stack([h_all, x_feat]))
    # ΔR² vs modelo h-only
    dr2, dr2_adj = delta_r2(y_all, X_h_only, x_feat)
    # ΔR² vs modelo base extendido (h + features ccovn existentes)
    dr2_ext, _ = delta_r2(y_all, X_base_ext, x_feat)

    row = {
        "feature":        feat,
        "coef_feature":   None,
        "t_stat_feature": None,
        "r2_modelo":      None,
        "r2_adj":         None,
        "delta_r2_vs_h":  round(dr2,     5) if np.isfinite(dr2)     else None,
        "delta_r2_vs_base": round(dr2_ext, 5) if np.isfinite(dr2_ext) else None,
        "n_obs":          None,
    }
    if res is not None:
        row.update({
            "coef_feature":   round(float(res["beta"][2]),   6),
            "t_stat_feature": round(float(res["t_stat"][2]), 2),
            "r2_modelo":      round(res["r2"],     4),
            "r2_adj":         round(res["r2_adj"], 4),
            "n_obs":          res["n"],
        })
    rows_ols.append(row)

df_ols = pd.DataFrame(rows_ols)

# R² del modelo base como referencia
res_h    = ols(y_all, X_h_only)
res_base = ols(y_all, X_base_ext)
r2_h_ref   = round(res_h["r2"],    4) if res_h    else None
r2_base_ref= round(res_base["r2"], 4) if res_base else None
print(f"    R² modelo h-only      : {r2_h_ref}")
print(f"    R² modelo base (h+ccovn existentes): {r2_base_ref}")
for r in rows_ols:
    print(f"    {r['feature']:<35} ΔR²(vs h)={r['delta_r2_vs_h']}  "
          f"ΔR²(vs base)={r['delta_r2_vs_base']}  t={r['t_stat_feature']}")

###############################################################################
# ANÁLISIS 3 — CORRELACIÓN ROLLING EN EL TIEMPO
###############################################################################

print("  [3/3] Correlación rolling en el tiempo ...")

# Usar h=2 (horizonte más informativo) para la rolling correlation
df_h2    = df_full[df_full["h"] == 2].copy().sort_values("fecha_t").reset_index(drop=True)
y_h2     = df_h2["target"].to_numpy(dtype=float)
fechas_h2 = pd.to_datetime(df_h2["fecha_t"])

rows_rolling = []
for feat in feats_disp:
    x_h2 = df_h2[feat].to_numpy(dtype=float) if feat in df_h2.columns else None
    if x_h2 is None:
        continue
    # Rolling window Pearson: manual sobre ventana deslizante
    r_series = np.full(len(y_h2), np.nan)
    for i in range(ROLLING_WINDOW, len(y_h2)):
        sl = slice(i - ROLLING_WINDOW, i)
        r, _, n = pearson(x_h2[sl], y_h2[sl])
        if n >= MIN_OBS:
            r_series[i] = r
    sr = pd.Series(r_series, index=fechas_h2)
    rows_rolling.append({
        "feature": feat,
        "series":  sr,
        "mean_r":  round(np.nanmean(r_series), 4),
        "std_r":   round(np.nanstd(r_series),  4),
        "pct_positivo": round(np.nanmean(r_series > 0) * 100, 1),
    })

df_rolling_meta = pd.DataFrame([
    {k: v for k, v in r.items() if k != "series"}
    for r in rows_rolling
])

###############################################################################
# GRÁFICOS
###############################################################################

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# ── Plot 1: Pearson + Spearman por bucket ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
fig.suptitle("Correlación con target por bucket de horizonte h", fontweight="bold")

for ax, metrica in zip(axes, ["pearson_r", "spearman_r"]):
    pivot = df_buckets.pivot(index="bucket", columns="feature", values=metrica)
    buckets_order = list(H_BUCKETS.keys())
    pivot = pivot.reindex(buckets_order)
    x      = np.arange(len(pivot))
    width  = 0.8 / max(len(feats_disp), 1)
    for i, feat in enumerate(feats_disp):
        if feat not in pivot.columns:
            continue
        vals = pivot[feat].to_numpy(dtype=float)
        bars = ax.bar(x + i * width - (len(feats_disp) - 1) * width / 2,
                      vals, width * 0.9, label=feat, color=COLORS[i % len(COLORS)],
                      alpha=0.85)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f"{v:.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title(metrica.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Correlación r")
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
p1 = DIR_OUTPUT / "00_corr_por_h.png"
fig.savefig(p1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\n  Guardado: {p1.name}")

# ── Plot 2: Pearson por cada h individual (2..75) ────────────────────────────
h_vals = sorted(df_full["h"].unique())
fig, ax = plt.subplots(figsize=(13, 4))
fig.suptitle("Correlación Pearson vs target  ·  por horizonte h individual", fontweight="bold")

for i, feat in enumerate(feats_disp):
    x_all = df_full[feat].to_numpy(dtype=float)
    rs = []
    for hv in h_vals:
        mh = h_all == hv
        r, _, _ = pearson(x_all[mh], y_all[mh])
        rs.append(r)
    rs = np.array(rs, dtype=float)
    # LOESS-like smooth: rolling mean 5h
    rs_s = pd.Series(rs).rolling(5, center=True, min_periods=2).mean().to_numpy()
    ax.plot(h_vals, rs_s, color=COLORS[i % len(COLORS)], linewidth=1.8,
            label=feat)
    ax.fill_between(h_vals, rs, alpha=0.08, color=COLORS[i % len(COLORS)])

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Horizonte h (días hábiles)")
ax.set_ylabel("Pearson r (suavizado 5h)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(min(h_vals), max(h_vals))
# Separadores de buckets
for _, (h_lo, h_hi) in H_BUCKETS.items():
    ax.axvline(h_hi + 0.5, color="gray", linewidth=0.5, linestyle=":")

plt.tight_layout()
p2 = DIR_OUTPUT / "01_corr_lineal_h.png"
fig.savefig(p2, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Guardado: {p2.name}")

# ── Plot 3: Rolling correlation en el tiempo ─────────────────────────────────
if rows_rolling:
    fig, axes = plt.subplots(len(rows_rolling), 1,
                             figsize=(13, 3 * len(rows_rolling)), sharex=True)
    if len(rows_rolling) == 1:
        axes = [axes]
    fig.suptitle(f"Correlación rolling {ROLLING_WINDOW}d con target  ·  h=2",
                 fontweight="bold")

    for ax, row in zip(axes, rows_rolling):
        sr = row["series"].dropna()
        ax.plot(sr.index, sr.values, color=COLORS[rows_rolling.index(row) % len(COLORS)],
                linewidth=1.2)
        ax.axhline(0,              color="black", linewidth=0.7, linestyle="--")
        ax.axhline(row["mean_r"],  color="gray",  linewidth=0.8, linestyle="-.",
                   label=f"media={row['mean_r']:.3f}")
        ax.fill_between(sr.index, sr.values, 0,
                        where=(sr.values > 0), alpha=0.12, color="green")
        ax.fill_between(sr.index, sr.values, 0,
                        where=(sr.values < 0), alpha=0.12, color="red")
        ax.set_title(f"{row['feature']}  "
                     f"(mean_r={row['mean_r']:.3f}, std={row['std_r']:.3f}, "
                     f"{row['pct_positivo']:.0f}% positivo)", fontsize=9)
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Fecha")
    plt.tight_layout()
    p3 = DIR_OUTPUT / "02_rolling_corr.png"
    fig.savefig(p3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {p3.name}")

###############################################################################
# EXPORTAR EXCEL
###############################################################################

print("\n  Exportando Excel ...")
RUTA_EXCEL = DIR_OUTPUT / "resultados_ccovn.xlsx"

try:
    with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:

        # Hoja 1: correlaciones por bucket
        df_buckets.to_excel(writer, sheet_name="Correlacion_por_h", index=False)

        # Hoja 2: OLS
        df_ols.to_excel(writer, sheet_name="OLS_univariado", index=False)

        # Hoja 3: rolling meta
        if not df_rolling_meta.empty:
            df_rolling_meta.to_excel(writer, sheet_name="Rolling_estabilidad", index=False)

        # Hoja 4: referencia R² modelos base
        pd.DataFrame([
            {"modelo": "h-only",              "r2": r2_h_ref},
            {"modelo": "h + ccovn existentes","r2": r2_base_ref},
        ]).to_excel(writer, sheet_name="Referencia_R2", index=False)

    print(f"  Guardado: {RUTA_EXCEL.name}")
except Exception as e:
    print(f"  AVISO: no se pudo exportar Excel — {e}")

###############################################################################
# RESUMEN CONSOLA
###############################################################################

print("\n" + "=" * 65)
print("  RESUMEN")
print("=" * 65)
print(f"  Referencia  R²(h-only)={r2_h_ref}  "
      f"R²(h+base_ccovn)={r2_base_ref}")
print()
for r in rows_ols:
    t = r["t_stat_feature"]
    sig = "***" if t and abs(t) > 3.3 else ("**" if t and abs(t) > 2.6
          else ("*" if t and abs(t) > 2.0 else "  "))
    print(f"  {r['feature']:<35} "
          f"t={str(r['t_stat_feature']):>6}  {sig}  "
          f"ΔR²(vs h)={str(r['delta_r2_vs_h']):>8}  "
          f"ΔR²(vs base)={str(r['delta_r2_vs_base']):>8}")

print(f"\n  Todos los archivos en: {DIR_OUTPUT}")
print("=" * 65)
