# -*- coding: utf-8 -*-
"""
aux_comparar_cv4_configs.py
===========================
Comparativa de las 4 configuraciones cv4 DIRECT:
    fold_exp_0.5  · fold_exp_1  · fold_roll_0.5  · fold_roll_1

Lee el parquet metricas_SISTEMA_*.parquet de cada carpeta y genera:
  1. Tabla resumen global (RMSE · CRPS · Cobertura 90%/98% · Winkler)
  2. Tabla por grupo de horizonte (muy_corto / corto / medio / largo)
  3. Cobertura 90% por fold × config (detección de drift)
  4. Calibración (hit-rate) por cuantil × config
  5. Plots: comparativa de métricas clave y cobertura por fold
  6. Excel consolidado en DIR_OUT / "comparativo_configs.xlsx"
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"
DIR_OUT      = DIR_WFCV / "comparativo_cv4_configs"
DIR_OUT.mkdir(parents=True, exist_ok=True)

BANCO = "SISTEMA"

CONFIGS = {
    "exp_0.5" : DIR_WFCV / "fold_exp_0.5",
    "exp_1"   : DIR_WFCV / "fold_exp_1",
    "roll_0.5": DIR_WFCV / "fold_roll_0.5",
    "roll_1"  : DIR_WFCV / "fold_roll_1",
}

H_GRUPOS = [
    ("muy_corto", 2,  5),
    ("corto",     6, 20),
    ("medio",    21, 50),
    ("largo",    51, 75),
]

CALIB_TAUS = [1, 5, 40, 50, 60, 95, 99]

COLORS = {
    "exp_0.5" : "#0F766E",   # teal — baseline recomendado
    "exp_1"   : "#1E3A5F",   # navy
    "roll_0.5": "#D97706",   # amber
    "roll_1"  : "#9333EA",   # purple
}

# ── 1. Cargar parquets ─────────────────────────────────────────────────────────
def cargar_parquet(cfg: str, path: Path) -> pd.DataFrame | None:
    candidatos = sorted(path.glob(f"metricas_{BANCO}_*.parquet"))
    if not candidatos:
        print(f"[WARN] Sin métricas en {path.name} — omitiendo")
        return None
    df = pd.read_parquet(candidatos[-1])
    df["config"] = cfg
    return df


dfs = {}
for cfg, path in CONFIGS.items():
    df = cargar_parquet(cfg, path)
    if df is not None:
        dfs[cfg] = df

if not dfs:
    raise RuntimeError("No se encontró ningún parquet de métricas.")

datos = pd.concat(dfs.values(), ignore_index=True)
print(f"\n[INFO] Columnas disponibles:\n  {sorted(datos.columns.tolist())}\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _h_grupo(h_series: pd.Series) -> pd.Categorical:
    bins   = [1, 5, 20, 50, 75]
    labels = ["muy_corto", "corto", "medio", "largo"]
    return pd.cut(h_series, bins=bins, labels=labels)


datos["h_grupo"] = _h_grupo(datos["h"])


def _fmt_mm(x):
    return f"{x/1e6:,.0f} MM" if pd.notna(x) else "—"

def _fmt_pct(x):
    return f"{x:.1%}" if pd.notna(x) else "—"


# ── 2. Tabla resumen global ────────────────────────────────────────────────────
print("=" * 70)
print("TABLA 1 — RESUMEN GLOBAL POR CONFIGURACIÓN")
print("=" * 70)

METRICS_GLOBAL = []
for col, fmt, label in [
    ("rmse",         _fmt_mm,  "RMSE"),
    ("crps",         _fmt_mm,  "CRPS"),
    ("coverage_90",  _fmt_pct, "Cobertura 90%"),
    ("coverage_98",  _fmt_pct, "Cobertura 98%"),
    ("val_coverage_90", _fmt_pct, "VAL Cob 90%"),
    ("winkler_90",   _fmt_mm,  "Winkler 90%"),
    ("pinball_q50",  _fmt_mm,  "Pinball Q50"),
]:
    if col not in datos.columns:
        continue
    row = {"Métrica": label}
    for cfg in CONFIGS:
        if cfg not in dfs:
            row[cfg] = "—"
            continue
        val = dfs[cfg][col].mean() if col in dfs[cfg].columns else np.nan
        row[cfg] = fmt(val)
    METRICS_GLOBAL.append(row)

df_global = pd.DataFrame(METRICS_GLOBAL)
print(df_global.to_string(index=False))


# ── 3. Tabla por grupo de horizonte ───────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLA 2 — RMSE POR GRUPO HORIZONTE")
print("=" * 70)

rmse_grp = datos.groupby(["config", "h_grupo"], observed=True)["rmse"].mean().unstack("config")
if not rmse_grp.empty:
    print((rmse_grp / 1e6).round(3).to_string())

print("\n" + "=" * 70)
print("TABLA 3 — COBERTURA 90% POR GRUPO HORIZONTE")
print("=" * 70)

if "coverage_90" in datos.columns:
    cob_grp = datos.groupby(["config", "h_grupo"], observed=True)["coverage_90"].mean().unstack("config")
    print(cob_grp.applymap(_fmt_pct).to_string())


# ── 4. Cobertura 90% por fold × config ────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLA 4 — COBERTURA 90% POR FOLD (media sobre todos los h)")
print("=" * 70)

if "coverage_90" in datos.columns and "fold" in datos.columns:
    cob_fold = datos.groupby(["config", "fold"])["coverage_90"].mean().unstack("config")
    print(cob_fold.applymap(_fmt_pct).to_string())


# ── 5. Calibración por cuantil ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLA 5 — CALIBRACIÓN HIT-RATE POR CUANTIL (global)")
print("=" * 70)

calib_rows = []
for tau in CALIB_TAUS:
    col = f"calib_q{tau:02d}"
    if col not in datos.columns:
        continue
    row = {"Cuantil": f"Q{tau:02d}", "Nominal": f"{tau/100:.0%}"}
    for cfg in CONFIGS:
        if cfg not in dfs:
            row[cfg] = "—"
            continue
        val = dfs[cfg][col].mean() if col in dfs[cfg].columns else np.nan
        row[cfg] = _fmt_pct(val)
    calib_rows.append(row)

if calib_rows:
    print(pd.DataFrame(calib_rows).to_string(index=False))


# ── 6. Plots ───────────────────────────────────────────────────────────────────
cfgs_ok = [c for c in CONFIGS if c in dfs]

# 6a. RMSE por horizonte (media sobre folds)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Comparativa cv4 — RMSE, CRPS, Cobertura 90% y Calibración Q95",
             fontsize=13, fontweight="bold")

ax = axes[0, 0]
for cfg in cfgs_ok:
    df_ = dfs[cfg]
    if "rmse" not in df_.columns:
        continue
    m = df_.groupby("h")["rmse"].mean() / 1e6
    ax.plot(m.index, m.values, color=COLORS[cfg], label=cfg, linewidth=1.8)
ax.set_title("RMSE por horizonte (MM USD)", fontsize=11)
ax.set_xlabel("h (días hábiles)")
ax.set_ylabel("RMSE (MM USD)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# 6b. CRPS por horizonte
ax = axes[0, 1]
for cfg in cfgs_ok:
    df_ = dfs[cfg]
    if "crps" not in df_.columns:
        continue
    m = df_.groupby("h")["crps"].mean() / 1e6
    ax.plot(m.index, m.values, color=COLORS[cfg], label=cfg, linewidth=1.8)
ax.set_title("CRPS por horizonte (MM USD)", fontsize=11)
ax.set_xlabel("h (días hábiles)")
ax.set_ylabel("CRPS (MM USD)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6c. Cobertura 90% por horizonte
ax = axes[1, 0]
if "coverage_90" in datos.columns:
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Objetivo 90%")
    for cfg in cfgs_ok:
        df_ = dfs[cfg]
        if "coverage_90" not in df_.columns:
            continue
        m = df_.groupby("h")["coverage_90"].mean()
        ax.plot(m.index, m.values, color=COLORS[cfg], label=cfg, linewidth=1.8)
    ax.set_title("Cobertura 90% [Q05–Q95] por horizonte", fontsize=11)
    ax.set_xlabel("h (días hábiles)")
    ax.set_ylabel("Cobertura empírica")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0.6, 1.0)

# 6d. Cobertura 90% por fold (barras)
ax = axes[1, 1]
if "coverage_90" in datos.columns and "fold" in datos.columns:
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, alpha=0.5)
    cob_fold_num = datos.groupby(["config", "fold"])["coverage_90"].mean().unstack("config")
    folds = sorted(cob_fold_num.index)
    n_cfg = len(cfgs_ok)
    width = 0.18
    x = np.arange(len(folds))
    for i, cfg in enumerate(cfgs_ok):
        if cfg not in cob_fold_num.columns:
            continue
        vals = cob_fold_num[cfg].reindex(folds).values
        ax.bar(x + i * width - (n_cfg - 1) * width / 2,
               vals, width=width, color=COLORS[cfg], label=cfg, alpha=0.85)
    ax.set_title("Cobertura 90% por fold", fontsize=11)
    ax.set_xlabel("Fold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
ruta_plot = DIR_OUT / f"comparativo_metricas_{BANCO}.png"
fig.savefig(ruta_plot, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] Plot guardado: {ruta_plot}")

# 6e. Calibración por cuantil — scatter nominal vs empírico
calib_cols_ok = [f"calib_q{t:02d}" for t in CALIB_TAUS
                 if f"calib_q{t:02d}" in datos.columns]
if calib_cols_ok:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Calibración perfecta")
    for cfg in cfgs_ok:
        df_ = dfs[cfg]
        nominals, empiricals = [], []
        for tau in CALIB_TAUS:
            col = f"calib_q{tau:02d}"
            if col not in df_.columns:
                continue
            nominals.append(tau / 100)
            empiricals.append(df_[col].mean())
        if nominals:
            ax.plot(nominals, empiricals, "o-", color=COLORS[cfg],
                    label=cfg, linewidth=1.8, markersize=6)
    ax.set_title("Calibración: nominal vs empírico por configuración", fontsize=11)
    ax.set_xlabel("τ nominal")
    ax.set_ylabel("Hit-rate empírico")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ruta_calib = DIR_OUT / f"comparativo_calibracion_{BANCO}.png"
    fig.savefig(ruta_calib, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot calibración: {ruta_calib}")


# ── 7. Excel consolidado ───────────────────────────────────────────────────────
ruta_excel = DIR_OUT / f"comparativo_configs_{BANCO}.xlsx"
with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
    # Hoja 1: resumen global
    df_global.to_excel(writer, sheet_name="Resumen_global", index=False)

    # Hoja 2: RMSE por grupo
    if "rmse" in datos.columns:
        rmse_grp2 = datos.groupby(["h_grupo", "config"], observed=True)["rmse"].mean().unstack("config")
        rmse_grp2.to_excel(writer, sheet_name="RMSE_por_grupo")

    # Hoja 3: cobertura 90% por grupo
    if "coverage_90" in datos.columns:
        cob_grp2 = datos.groupby(["h_grupo", "config"], observed=True)["coverage_90"].mean().unstack("config")
        cob_grp2.to_excel(writer, sheet_name="Cob90_por_grupo")

    # Hoja 4: cobertura 90% por fold
    if "coverage_90" in datos.columns and "fold" in datos.columns:
        cob_fold2 = datos.groupby(["fold", "config"])["coverage_90"].mean().unstack("config")
        cob_fold2.to_excel(writer, sheet_name="Cob90_por_fold")

    # Hoja 5: calibración
    if calib_rows:
        pd.DataFrame(calib_rows).to_excel(writer, sheet_name="Calibracion", index=False)

    # Hoja 6: serie h completa (RMSE y cobertura)
    for col in ["rmse", "coverage_90", "crps", "winkler_90"]:
        if col not in datos.columns:
            continue
        pivot = datos.groupby(["h", "config"])[col].mean().unstack("config")
        pivot.to_excel(writer, sheet_name=f"{col[:15]}_por_h")

print(f"[OK] Excel guardado: {ruta_excel}\n")

# ── 8. Tablas estadísticas descriptivas ───────────────────────────────────────
print("=" * 70)
print("TABLA 6 — ESTADÍSTICOS DESCRIPTIVOS POR CONFIGURACIÓN")
print("=" * 70)

stat_cols = [c for c in ["rmse", "crps", "coverage_90", "coverage_98",
                          "pinball_q50", "winkler_90"]
             if c in datos.columns]

for col in stat_cols:
    print(f"\n  {col.upper()}")
    tbl = datos.groupby("config")[col].agg(["mean", "std", "min", "max"])
    tbl.columns = ["Media", "Desv.Std", "Mín", "Máx"]
    if col in ("coverage_90", "coverage_98"):
        print(tbl.applymap(_fmt_pct).to_string())
    else:
        print((tbl / 1e6).round(3).rename(columns={
            "Media": "Media (MM)", "Desv.Std": "Std (MM)",
            "Mín": "Mín (MM)", "Máx": "Máx (MM)"
        }).to_string())

# Agregar hoja de estadísticos al Excel existente
with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    for col in stat_cols:
        tbl = datos.groupby("config")[col].agg(["mean", "std", "min", "max"])
        tbl.columns = ["Media", "Desv_Std", "Min", "Max"]
        tbl.to_excel(writer, sheet_name=f"stats_{col[:14]}")

    # Tabla pivot: h_grupo × config con media ± std
    if "rmse" in datos.columns:
        mean_ = datos.groupby(["h_grupo", "config"], observed=True)["rmse"].mean().unstack("config")
        std_  = datos.groupby(["h_grupo", "config"], observed=True)["rmse"].std().unstack("config")
        combined = mean_.round(0).astype(str) + " ± " + std_.round(0).astype(str)
        combined.to_excel(writer, sheet_name="RMSE_mean_std")

print(f"\n[OK] Estadísticos descriptivos añadidos a {ruta_excel}")

# ── 9. Ranking final ───────────────────────────────────────────────────────────
print("=" * 70)
print("RANKING FINAL — PUNTUACIÓN COMPUESTA (menor es mejor)")
print("=" * 70)

rank_metrics = ["rmse", "crps", "coverage_90", "winkler_90"]
scores = {}
for cfg in cfgs_ok:
    df_ = dfs[cfg]
    score = 0.0
    parts = []
    for col in rank_metrics:
        if col not in df_.columns:
            continue
        val = df_[col].mean()
        # normalizar vs media del grupo para hacer comparable
        all_vals = [dfs[c][col].mean() for c in cfgs_ok if col in dfs[c].columns]
        ref = np.mean(all_vals)
        if ref == 0:
            continue
        if col == "coverage_90":
            # distancia al objetivo 90%
            normalized = abs(val - 0.90) / 0.90
        else:
            normalized = val / ref
        score += normalized
        parts.append(f"{col}={normalized:.3f}")
    scores[cfg] = (score, parts)

sorted_configs = sorted(scores.items(), key=lambda x: x[1][0])
for rank, (cfg, (score, parts)) in enumerate(sorted_configs, 1):
    marker = " ★ RECOMENDADO" if rank == 1 else ""
    print(f"  #{rank} {cfg:12s}  score={score:.4f}{marker}")
    print(f"       {' | '.join(parts)}")
print()
