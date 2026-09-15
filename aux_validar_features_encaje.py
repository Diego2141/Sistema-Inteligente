# -*- coding: utf-8 -*-
"""
aux_validar_features_encaje.py
================================
Valida estadísticamente los features de estrategia sobreencaje BBVA antes
de incorporarlos al modelo XGBoost.

Fuente: bbva_encaje_features.xlsx (output de aux_encaje_2.py)
  Hoja 'Datos' — serie diaria BBVA

Features propuestos (todos con lag=1 día para evitar leakage):
  1. saldo_vs_exigible_acum   — banco adelantado al ritmo requerido del mes
  2. urgencia                 — saldo adelantado × 1/días_restantes
  3. es_fin_trimestre         — mes ∈ {3,6,9,12} (patrón trimestral 2025)
  4. estrategia_bbva_lag1     — ¿estrategia aplicada el mes anterior?
  5. estrategia_bbva_lag3     — ¿estrategia aplicada hace 3 meses?
  6. n_estrat_12m             — meses con estrategia en los últimos 12

Target: retiro_neto BBVA, simulando h días adelante (h = 1,2,5,10,20,40,75)

Análisis:
  Bloque 1 — Spearman por horizonte h (todas las fechas)
  Bloque 2 — Spearman condicional: pre-2024 vs 2024+
  Bloque 3 — ΔR² incremental sobre base diaria (2024+, h=1 y h=5)
  Bloque 4 — Rolling 252d para los 2 features más importantes

Outputs → DIR_OUT /
  00_corr_por_h.png
  01_corr_por_periodo.png
  02_delta_r2.png
  03_rolling_corr.png
  resultados_encaje_features.xlsx
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

BASE       = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_XLSX  = BASE / "2. Output" / "encaje_bbva" / "bbva_encaje_features.xlsx"
DIR_OUT    = BASE / "2. Output" / "validacion_features_encaje"
DIR_OUT.mkdir(parents=True, exist_ok=True)

UMBRAL_SALDO   = 0.50          # mismo que aux_encaje_2.py
CORTE_RECIENTE = "2024-01-01"  # divide histórco vs reciente
HORIZONTES     = [1, 2, 5, 10, 20, 40, 75]
ROLLING_WIN    = 252           # ~1 año hábil
MIN_OBS        = 30

FEATURES = [
    "saldo_vs_exigible_acum_lag1",
    "urgencia_lag1",
    "es_fin_trimestre",
    "estrategia_bbva_lag1",
    "estrategia_bbva_lag3",
    "n_estrat_12m",
]

LABELS = {
    "saldo_vs_exigible_acum_lag1": "Saldo vs exigible acum (t-1)",
    "urgencia_lag1":               "Urgencia = saldo/días rest (t-1)",
    "es_fin_trimestre":            "Fin de trimestre (cal.)",
    "estrategia_bbva_lag1":        "Estrategia mes anterior",
    "estrategia_bbva_lag3":        "Estrategia hace 3 meses",
    "n_estrat_12m":                "Nº meses estrat. últ. 12m",
}

# Features base para ΔR² (ya presentes en el comportamiento diario)
FEATURES_BASE = ["dia_mes", "dias_restantes", "es_fin_trimestre"]

###############################################################################
# FUNCIONES ESTADÍSTICAS (pure numpy, sin scipy)
###############################################################################

def _mask(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a.astype(float))
    return m


def spearman(x, y):
    """Spearman r, t-stat, n. Devuelve (nan, nan, 0) si n < MIN_OBS."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = _mask(x, y)
    n = int(m.sum())
    if n < MIN_OBS:
        return np.nan, np.nan, n
    def _rank(v):
        o = np.argsort(v); r = np.empty(n, float)
        r[o] = np.arange(n)
        # ties
        i = 0
        while i < n:
            j = i + 1
            while j < n and v[o[j]] == v[o[i]]:
                j += 1
            r[o[i:j]] = r[o[i:j]].mean()
            i = j
        return r - (n - 1) / 2
    rx = _rank(x[m]); ry = _rank(y[m])
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    r = (rx @ ry) / denom if denom > 0 else np.nan
    t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-15)) if np.isfinite(r) else np.nan
    return float(r), float(t), n


def ols(y, X):
    """OLS (intercepto agregado internamente). Retorna dict o None."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    m = _mask(y) & np.all(np.isfinite(X), axis=1)
    n, k = int(m.sum()), X.shape[1] + 1
    if n < k + 5:
        return None
    y_, Xb = y[m], np.column_stack([np.ones(n), X[m]])
    try:
        beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y_)
    except np.linalg.LinAlgError:
        return None
    resid = y_ - Xb @ beta
    ss_res, ss_tot = resid @ resid, ((y_ - y_.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return {"r2": r2, "r2_adj": r2_adj, "n": n}


def delta_r2(y, X_base, x_new):
    base = ols(y, X_base)
    aug  = ols(y, np.column_stack([X_base, np.asarray(x_new, float)]))
    if base is None or aug is None:
        return np.nan, np.nan
    return aug["r2"] - base["r2"], aug["r2_adj"] - base["r2_adj"]

###############################################################################
# CARGA Y PREPARACIÓN DE DATOS
###############################################################################

print("=" * 65)
print("  aux_validar_features_encaje")
print("=" * 65)

if not RUTA_XLSX.exists():
    raise FileNotFoundError(
        f"No encontrado: {RUTA_XLSX}\n"
        "Ejecutar aux_encaje_2.py primero."
    )

print(f"\n  Leyendo: {RUTA_XLSX.name}")
df = pd.read_excel(RUTA_XLSX, sheet_name="Datos")
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
print(f"  Período  : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"  Filas    : {len(df):,}")

# ── Features diarios ──────────────────────────────────────────────────────────

# 1. saldo_vs_exigible_acum: ¿cuánto está adelantado el banco respecto al ritmo?
#    = encaje acumulado / exigible acumulado - 1 → positivo = sobreencajando
df["saldo_vs_exigible_acum"] = (
    df["EncajeAcumMes"] / df["NecAcumMes"].replace(0, np.nan) - 1
)

# 2. urgencia: amplifica la señal al acercarse el fin de mes
#    alto saldo + pocos días = el banco tiene que retirar pronto
df["urgencia"] = df["saldo_vs_exigible_acum"] / df["dias_restantes"].clip(lower=1)

# 3. Fin de trimestre (feature de calendario, no requiere lag)
df["es_fin_trimestre"] = df["fecha"].dt.month.isin([3, 6, 9, 12]).astype(int)

# ── Features mensuales (estrategia) ──────────────────────────────────────────
df["_am"] = df["fecha"].dt.to_period("M")

def _estrat_mes(g):
    g    = g.sort_values("fecha")
    late = g.tail(5)
    ret  = late["retiro_neto"].sum() if len(late) else np.nan
    med  = g["encaje_ovn"].median()
    return pd.Series({
        "retiro_acum_5d_M": ret / 1e6 if np.isfinite(ret) else np.nan,
        "mediana_saldo_M":  med / 1e6 if np.isfinite(med) else np.nan,
        "estrategia":       (np.isfinite(ret) and np.isfinite(med)
                             and ret < -(UMBRAL_SALDO * med)),
    })

dm = df.groupby("_am").apply(_estrat_mes).reset_index()
dm = dm.sort_values("_am").reset_index(drop=True)

# Lags mensuales (shift sobre la serie ordenada)
dm["estrategia_bbva_lag1"] = dm["estrategia"].shift(1).astype("boolean")
dm["estrategia_bbva_lag3"] = dm["estrategia"].shift(3).astype("boolean")
dm["n_estrat_12m"]         = (dm["estrategia"].astype(float)
                               .rolling(12, min_periods=1).sum().shift(1))

# Unir features mensuales al dataframe diario
df = df.merge(
    dm[["_am", "estrategia_bbva_lag1", "estrategia_bbva_lag3", "n_estrat_12m"]],
    on="_am", how="left"
)

# ── Aplicar lag=1 día a features diarios ─────────────────────────────────────
df["saldo_vs_exigible_acum_lag1"] = df["saldo_vs_exigible_acum"].shift(1)
df["urgencia_lag1"]               = df["urgencia"].shift(1)

# Convertir booleans a float para estadísticos
for _c in ["estrategia_bbva_lag1", "estrategia_bbva_lag3"]:
    df[_c] = df[_c].astype(float)
df["n_estrat_12m"] = df["n_estrat_12m"].astype(float)

print(f"\n  Meses con estrategia: "
      f"{dm['estrategia'].sum()} / {len(dm)} "
      f"({dm['estrategia'].mean()*100:.0f}%)")
print(f"  Features disponibles: {[f for f in FEATURES if f in df.columns]}")

###############################################################################
# BLOQUE 1 — SPEARMAN POR HORIZONTE H
###############################################################################

print("\n── Bloque 1: Spearman por horizonte h ──────────────────────────────")
print(f"  {'Feature':<34} " + "  ".join(f"h={h:>3}" for h in HORIZONTES))
print(f"  {'-'*34} " + "  ".join("-" * 6 for _ in HORIZONTES))

rows_b1 = []
for feat in FEATURES:
    if feat not in df.columns:
        continue
    x = df[feat].values
    row = {"feature": feat}
    vals = []
    for h in HORIZONTES:
        tgt = df["retiro_neto"].shift(-h).values
        r, t, n = spearman(x, tgt)
        row[f"h{h}_r"] = round(r, 3) if np.isfinite(r) else np.nan
        row[f"h{h}_t"] = round(t, 2) if np.isfinite(t) else np.nan
        row[f"h{h}_n"] = n
        vals.append(f"{r:+.3f}" if np.isfinite(r) else "   nan")
    rows_b1.append(row)
    print(f"  {LABELS.get(feat, feat):<34} {'  '.join(vals)}")

df_b1 = pd.DataFrame(rows_b1)

###############################################################################
# BLOQUE 2 — SPEARMAN CONDICIONAL: PRE-2024 vs 2024+
###############################################################################

print("\n── Bloque 2: Spearman por período ──────────────────────────────────")
print(f"  {'Feature':<34} {'Pre-2024':>10}  {'2024+':>10}  {'Diferencia':>10}")
print(f"  {'-'*34} {'-'*10}  {'-'*10}  {'-'*10}")

mask_rec = df["fecha"] >= CORTE_RECIENTE
mask_his = ~mask_rec

rows_b2 = []
# Para el bloque 2 usamos h=1 (señal más inmediata)
tgt_h1 = df["retiro_neto"].shift(-1).values
for feat in FEATURES:
    if feat not in df.columns:
        continue
    x = df[feat].values
    r_his, _, n_his = spearman(x[mask_his], tgt_h1[mask_his])
    r_rec, _, n_rec = spearman(x[mask_rec], tgt_h1[mask_rec])
    dif = (r_rec - r_his) if (np.isfinite(r_rec) and np.isfinite(r_his)) else np.nan
    rows_b2.append({
        "feature": feat,
        "r_historico": round(r_his, 3) if np.isfinite(r_his) else np.nan,
        "n_historico": n_his,
        "r_reciente":  round(r_rec, 3) if np.isfinite(r_rec) else np.nan,
        "n_reciente":  n_rec,
        "diferencia":  round(dif, 3)   if np.isfinite(dif)   else np.nan,
    })
    r_h = f"{r_his:+.3f} (n={n_his})" if np.isfinite(r_his) else "     nan"
    r_r = f"{r_rec:+.3f} (n={n_rec})" if np.isfinite(r_rec) else "     nan"
    d   = f"{dif:+.3f}"               if np.isfinite(dif)   else "     nan"
    print(f"  {LABELS.get(feat, feat):<34} {r_h:>10}  {r_r:>10}  {d:>10}")

df_b2 = pd.DataFrame(rows_b2)

# También calculamos por múltiples h en período reciente
rows_b2_h = []
for feat in FEATURES:
    if feat not in df.columns:
        continue
    x_rec = df.loc[mask_rec, feat].values
    row = {"feature": feat, "periodo": "2024+"}
    for h in HORIZONTES:
        tgt = df["retiro_neto"].shift(-h).values
        r, t, n = spearman(x_rec, tgt[mask_rec])
        row[f"h{h}_r"] = round(r, 3) if np.isfinite(r) else np.nan
        row[f"h{h}_n"] = n
    rows_b2_h.append(row)
df_b2_h = pd.DataFrame(rows_b2_h)

###############################################################################
# BLOQUE 3 — ΔR² INCREMENTAL (2024+, h=1 y h=5)
###############################################################################

print("\n── Bloque 3: ΔR² incremental (2024+) ──────────────────────────────")
print(f"  {'Feature':<34} {'ΔR² h=1':>10}  {'ΔR² h=5':>10}  {'R²_aug h=1':>12}")
print(f"  {'-'*34} {'-'*10}  {'-'*10}  {'-'*12}")

df_rec = df[mask_rec].copy()
rows_b3 = []
for h_eval in [1, 5]:
    tgt_rec = df["retiro_neto"].shift(-h_eval).values[mask_rec]
    # Base: dia_mes + dias_restantes + es_fin_trimestre
    X_base = df_rec[FEATURES_BASE].values.astype(float)
    for feat in FEATURES:
        if feat not in df_rec.columns:
            continue
        if feat in FEATURES_BASE:
            continue
        x_new = df_rec[feat].values.astype(float)
        dr2, dr2_adj = delta_r2(tgt_rec, X_base, x_new)
        aug = ols(tgt_rec, np.column_stack([X_base, x_new]))
        r2_aug = aug["r2"] if aug else np.nan
        rows_b3.append({
            "feature": feat, "h": h_eval,
            "delta_r2":     round(dr2,    4) if np.isfinite(dr2)    else np.nan,
            "delta_r2_adj": round(dr2_adj,4) if np.isfinite(dr2_adj)else np.nan,
            "r2_aug":       round(r2_aug, 4) if np.isfinite(r2_aug) else np.nan,
        })

df_b3 = pd.DataFrame(rows_b3)
# Print h=1 vs h=5
for feat in FEATURES:
    if feat not in df_rec.columns or feat in FEATURES_BASE:
        continue
    r1 = df_b3[(df_b3.feature == feat) & (df_b3.h == 1)]
    r5 = df_b3[(df_b3.feature == feat) & (df_b3.h == 5)]
    dr2_1 = f"{r1['delta_r2'].iloc[0]:+.4f}"  if len(r1) else "     nan"
    dr2_5 = f"{r5['delta_r2'].iloc[0]:+.4f}"  if len(r5) else "     nan"
    r2a   = f"{r1['r2_aug'].iloc[0]:.4f}"     if len(r1) else "     nan"
    print(f"  {LABELS.get(feat, feat):<34} {dr2_1:>10}  {dr2_5:>10}  {r2a:>12}")

###############################################################################
# BLOQUE 4 — ROLLING 252d PARA LOS 2 FEATURES CLAVE
###############################################################################

print("\n── Bloque 4: Rolling 252d ───────────────────────────────────────────")
ROLLING_FEATS = ["saldo_vs_exigible_acum_lag1", "n_estrat_12m"]
tgt_h1_arr = df["retiro_neto"].shift(-1).values

roll_rows = []
for feat in ROLLING_FEATS:
    if feat not in df.columns:
        continue
    x = df[feat].values
    rs = []
    for i in range(len(df)):
        start = max(0, i - ROLLING_WIN + 1)
        xi = x[start:i+1]; yi = tgt_h1_arr[start:i+1]
        r, _, n = spearman(xi, yi)
        rs.append(r if n >= MIN_OBS else np.nan)
    df[f"_roll_{feat}"] = rs
    valid = [v for v in rs if np.isfinite(v)]
    print(f"  {feat:<38}: mean_r={np.nanmean(rs):+.3f}  "
          f"pct_neg={sum(v < 0 for v in valid)/max(len(valid),1)*100:.0f}%")
    roll_rows.append({"feature": feat, "mean_r": np.nanmean(rs),
                      "pct_neg": sum(v < 0 for v in valid)/max(len(valid),1)})

###############################################################################
# GRÁFICO 1 — HEATMAP SPEARMAN por h (todas las fechas + período reciente)
###############################################################################

plt.rcParams.update({"font.size": 9})
_feat_labels = [LABELS.get(f, f) for f in FEATURES if f in df.columns]
_feats_ok    = [f for f in FEATURES if f in df.columns]

fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(_feats_ok) * 0.6 + 2)))
fig.suptitle("Spearman r — features encaje BBVA vs retiro_neto[t+h]\n"
             "(negativo = predice más retiro)",
             fontweight="bold", fontsize=11)

for ax_idx, (mask_title, mask_use) in enumerate([
    ("Todas las fechas",   np.ones(len(df), bool)),
    (f"Desde {CORTE_RECIENTE}", mask_rec),
]):
    mat = np.full((len(_feats_ok), len(HORIZONTES)), np.nan)
    for i, feat in enumerate(_feats_ok):
        for j, h in enumerate(HORIZONTES):
            tgt = df["retiro_neto"].shift(-h).values
            r, _, n = spearman(df[feat].values[mask_use], tgt[mask_use])
            mat[i, j] = r

    ax = axes[ax_idx]
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(HORIZONTES)))
    ax.set_xticklabels([f"h={h}" for h in HORIZONTES], fontsize=8)
    ax.set_yticks(range(len(_feats_ok)))
    ax.set_yticklabels(_feat_labels, fontsize=8)
    ax.set_title(mask_title)
    for i in range(len(_feats_ok)):
        for j in range(len(HORIZONTES)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(v) > 0.3 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
p1 = DIR_OUT / "00_corr_por_h.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Guardado: {p1.name}")

###############################################################################
# GRÁFICO 2 — BARRAS: PRE-2024 vs 2024+ (h=1)
###############################################################################

fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(_feats_ok) * 0.55 + 2)))
fig.suptitle(f"Spearman r con retiro_neto[t+1]\n"
             f"Período histórico vs reciente — señal creciente valida el cambio de régimen",
             fontweight="bold", fontsize=10)

for ax_idx, h_show in enumerate([1, 5]):
    ax = axes[ax_idx]
    rs_his, rs_rec = [], []
    for feat in _feats_ok:
        x = df[feat].values
        tgt = df["retiro_neto"].shift(-h_show).values
        r_h, _, _ = spearman(x[mask_his], tgt[mask_his])
        r_r, _, _ = spearman(x[mask_rec], tgt[mask_rec])
        rs_his.append(r_h if np.isfinite(r_h) else 0)
        rs_rec.append(r_r if np.isfinite(r_r) else 0)

    y = np.arange(len(_feats_ok))
    bw = 0.35
    ax.barh(y + bw/2, rs_his, bw, color="#90A4AE", label=f"Pre-{CORTE_RECIENTE[:4]}", alpha=0.85)
    ax.barh(y - bw/2, rs_rec, bw, color="#E53935", label=f"Desde {CORTE_RECIENTE[:4]}", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(_feat_labels, fontsize=8)
    ax.set_xlabel("Spearman r")
    ax.set_title(f"h = {h_show}")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.6, 0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))

plt.tight_layout()
p2 = DIR_OUT / "01_corr_por_periodo.png"
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Guardado: {p2.name}")

###############################################################################
# GRÁFICO 3 — ΔR² INCREMENTAL (2024+)
###############################################################################

if not df_b3.empty:
    feats_b3 = [f for f in _feats_ok if f not in FEATURES_BASE]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, len(feats_b3) * 0.6 + 2)))
    fig.suptitle(f"ΔR² incremental sobre base [dia_mes, dias_restantes, es_fin_trimestre]\n"
                 f"Período 2024+ — mayor barra = más ganancia explicativa",
                 fontweight="bold", fontsize=10)

    for ax_idx, h_show in enumerate([1, 5]):
        ax    = axes[ax_idx]
        sub   = df_b3[df_b3["h"] == h_show].set_index("feature")
        vals  = [sub.loc[f, "delta_r2"] if f in sub.index else np.nan
                 for f in feats_b3]
        lbls  = [LABELS.get(f, f) for f in feats_b3]
        cols  = ["#E53935" if (v is not None and np.isfinite(v) and v > 0)
                 else "#90A4AE" for v in vals]
        y = np.arange(len(feats_b3))
        ax.barh(y, vals, color=cols, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(lbls, fontsize=8)
        ax.set_xlabel("ΔR²")
        ax.set_title(f"h = {h_show}")

    plt.tight_layout()
    p3 = DIR_OUT / "02_delta_r2.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {p3.name}")

###############################################################################
# GRÁFICO 4 — ROLLING 252d
###############################################################################

roll_cols = [f"_roll_{f}" for f in ROLLING_FEATS if f"_roll_{f}" in df.columns]
if roll_cols:
    fig, axes = plt.subplots(len(roll_cols), 1,
                             figsize=(15, 4 * len(roll_cols)), sharex=True)
    if len(roll_cols) == 1:
        axes = [axes]
    fig.suptitle(f"Correlación rolling {ROLLING_WIN}d (Spearman) con retiro_neto[t+1]\n"
                 "La señal creciente (más negativa) confirma el cambio de régimen",
                 fontweight="bold")

    for ax, feat in zip(axes, ROLLING_FEATS):
        col = f"_roll_{feat}"
        if col not in df.columns:
            continue
        vals  = df[col].values
        pct_n = sum(v < 0 for v in vals if np.isfinite(v)) / max(sum(np.isfinite(v) for v in vals), 1)
        ax.plot(df["fecha"], vals, color="#1565C0", lw=1.2)
        ax.fill_between(df["fecha"], vals, 0,
                        where=np.array(vals, dtype=float) < 0,
                        color="#E53935", alpha=0.25, label="Período negativo")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(pd.Timestamp(CORTE_RECIENTE), color="orange",
                   lw=1.2, ls="--", label=f"Corte {CORTE_RECIENTE[:4]}")
        ax.axhline(np.nanmean(vals), color="gray", lw=0.8, ls="-.",
                   label=f"Media = {np.nanmean(vals):+.3f}")
        ax.set_ylabel("Spearman r")
        ax.set_title(f"{LABELS.get(feat, feat)}  (% ventanas negativas: {pct_n*100:.0f}%)")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.7, 0.7)

    plt.tight_layout()
    p4 = DIR_OUT / "03_rolling_corr.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {p4.name}")

###############################################################################
# EXPORT EXCEL
###############################################################################

print("\n  Exportando Excel ...")

# Datos diarios completos (para inspección manual)
_cols_export = ["fecha", "dia_mes", "dias_restantes",
                "encaje_ovn", "NecAcumMes", "EncajeAcumMes",
                "retiro_neto", "Avance", "brecha_90"] + FEATURES

df_export = df[[c for c in _cols_export if c in df.columns]].copy()

# Agregar target h=1 y h=5 para facilitar validación manual
df_export["target_retiro_h1"] = df["retiro_neto"].shift(-1)
df_export["target_retiro_h5"] = df["retiro_neto"].shift(-5)

# Tabla de resumen mensual de estrategia
df_dm_export = dm[["_am", "retiro_acum_5d_M", "mediana_saldo_M", "estrategia",
                   "estrategia_bbva_lag1", "estrategia_bbva_lag3",
                   "n_estrat_12m"]].copy()
df_dm_export.rename(columns={"_am": "mes"}, inplace=True)

try:
    ruta_xlsx = DIR_OUT / "resultados_encaje_features.xlsx"
    with pd.ExcelWriter(ruta_xlsx, engine="openpyxl") as wr:
        df_export.to_excel(wr, sheet_name="Datos_diarios", index=False)
        df_dm_export.to_excel(wr, sheet_name="Resumen_mensual", index=False)
        df_b1.to_excel(wr, sheet_name="Corr_por_h", index=False)
        df_b2.to_excel(wr, sheet_name="Corr_por_periodo_h1", index=False)
        df_b2_h.to_excel(wr, sheet_name="Corr_periodo_rec_h", index=False)
        if not df_b3.empty:
            df_b3.to_excel(wr, sheet_name="Delta_R2_2024", index=False)
    print(f"  Guardado: {ruta_xlsx.name}")
    print(f"    Hoja 'Datos_diarios'       : {len(df_export):,} filas")
    print(f"    Hoja 'Resumen_mensual'     : {len(df_dm_export):,} filas")
    print(f"    Hoja 'Corr_por_h'          : {len(df_b1)} features × {len(HORIZONTES)} h")
    print(f"    Hoja 'Corr_por_periodo_h1' : pre-2024 vs 2024+")
    print(f"    Hoja 'Delta_R2_2024'       : ΔR² incremental 2024+")
except Exception as e:
    print(f"  AVISO: no se pudo exportar — {e}")

# Limpiar columnas temporales
df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

print(f"\n  Archivos en: {DIR_OUT}")
print("=" * 65)
