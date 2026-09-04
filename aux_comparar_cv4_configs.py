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

CALIB_TAUS = [1, 5, 40, 50, 60, 95, 99]

COLORS = {
    "exp_0.5" : "#0F766E",   # teal — baseline recomendado
    "exp_1"   : "#1E3A5F",   # navy
    "roll_0.5": "#D97706",   # amber
    "roll_1"  : "#9333EA",   # purple
}

# ── 1. Cargar parquets ─────────────────────────────────────────────────────────
def cargar_parquet(cfg: str, path: Path):
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
cfgs_ok = [c for c in CONFIGS if c in dfs]   # lista ordenada de configs disponibles
print(f"\n[INFO] Columnas disponibles:\n  {sorted(datos.columns.tolist())}\n")
print(f"[INFO] Configs cargadas: {cfgs_ok}\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _h_grupo(h_series):
    bins   = [1, 5, 20, 50, 75]
    labels = ["muy_corto", "corto", "medio", "largo"]
    return pd.cut(h_series, bins=bins, labels=labels)

def _h_grupo5(h_series):
    """5 grupos de horizonte para la tabla fold × h_grupo."""
    bins   = [1,  5, 15, 30, 50, 75]
    labels = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]
    return pd.cut(h_series, bins=bins, labels=labels)


datos["h_grupo"]  = _h_grupo(datos["h"])
datos["h_grupo5"] = _h_grupo5(datos["h"])


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


# ── 4b. Cobertura 90% por fold × grupo horizonte (una tabla por config) ───────
print("\n" + "=" * 70)
print("TABLA 4b — COBERTURA 90% TEST: fold × grupo horizonte (por config)")
print("=" * 70)

pivot_fold_hgrp = {}   # guardamos para HTML y Excel
if "coverage_90" in datos.columns and "fold" in datos.columns:
    for cfg in cfgs_ok:
        df_ = dfs[cfg]
        if "coverage_90" not in df_.columns:
            continue
        df_["h_grupo5"] = _h_grupo5(df_["h"])
        piv = (
            df_.groupby(["fold", "h_grupo5"], observed=True)["coverage_90"]
            .mean()
            .unstack("h_grupo5")
        )
        piv.index = [f"Fold {f}" for f in piv.index]
        piv.columns = [str(c) for c in piv.columns]
        pivot_fold_hgrp[cfg] = piv
        print(f"\n  [{cfg}]  Cobertura 90% TEST por fold × grupo horizonte")
        print(piv.applymap(_fmt_pct).to_string())

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

    # Hoja 4b: fold × h_grupo5 por config
    for cfg, piv in pivot_fold_hgrp.items():
        piv.to_excel(writer, sheet_name=f"Cob90_fold_h_{cfg[:10]}")

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
winner = sorted_configs[0][0] if sorted_configs else None
for rank, (cfg, (score, parts)) in enumerate(sorted_configs, 1):
    marker = " ★ RECOMENDADO" if rank == 1 else ""
    print(f"  #{rank} {cfg:12s}  score={score:.4f}{marker}")
    print(f"       {' | '.join(parts)}")
print()


# ── 10. Checklist de consistencia ─────────────────────────────────────────────
print("=" * 70)
print("CHECKLIST DE CONSISTENCIA — VALIDACIÓN DE TABLAS")
print("=" * 70)

checks = []

def _chk(label, ok, detalle=""):
    sym = "✓" if ok else "✗"
    checks.append((sym, label, detalle))
    print(f"  [{sym}] {label}")
    if detalle:
        print(f"       {detalle}")

# C1 — Todos los configs cargados
_chk("Configs cargadas",
     len(cfgs_ok) == len(CONFIGS),
     f"Esperados {len(CONFIGS)}, cargados {len(cfgs_ok)}: {cfgs_ok}")

# C2 — Columnas clave presentes
for col in ["rmse", "crps", "coverage_90", "coverage_98", "fold", "h"]:
    _chk(f"Columna '{col}' presente",
         col in datos.columns,
         f"Columnas disponibles: {[c for c in datos.columns if c in ['rmse','crps','coverage_90','coverage_98','fold','h','winkler_90']]}")

# C3 — Folds consistentes por config
if "fold" in datos.columns:
    for cfg in cfgs_ok:
        n_folds = dfs[cfg]["fold"].nunique()
        expected = 5 if "0.5" in cfg else 4
        _chk(f"{cfg}: nº folds = {n_folds}",
             n_folds == expected,
             f"Esperados {expected} (test={'0.5' if '0.5' in cfg else '1'} año)")

# C4 — Rango de h correcto
if "h" in datos.columns:
    h_min_obs = datos["h"].min()
    h_max_obs = datos["h"].max()
    _chk(f"Rango h = [{h_min_obs}, {h_max_obs}]",
         h_min_obs >= 2 and h_max_obs <= 75,
         f"Esperado h ∈ [2, 75]")

# C5 — Coverage_90 en rango plausible [0.5, 1.0]
if "coverage_90" in datos.columns:
    cov_min = datos["coverage_90"].min()
    cov_max = datos["coverage_90"].max()
    _chk("coverage_90 en rango [0.5, 1.0]",
         0.5 <= cov_min and cov_max <= 1.0,
         f"Observado: [{cov_min:.3f}, {cov_max:.3f}]")

# C6 — RMSE positivo y en rango razonable (100M–2000M USD)
if "rmse" in datos.columns:
    rmse_min = datos["rmse"].min()
    rmse_max = datos["rmse"].max()
    _chk("RMSE en rango plausible (100M–2000M USD)",
         1e8 <= rmse_min and rmse_max <= 2e9,
         f"Observado: [{rmse_min/1e6:.0f} MM, {rmse_max/1e6:.0f} MM]")

# C7 — Tabla resumen coincide con datos
if "coverage_90" in datos.columns:
    for cfg in cfgs_ok:
        val_tbl  = dfs[cfg]["coverage_90"].mean()
        val_pivot = datos[datos["config"] == cfg]["coverage_90"].mean()
        _chk(f"{cfg}: cob90 resumen == pivot",
             abs(val_tbl - val_pivot) < 1e-9,
             f"Tabla={val_tbl:.4f}, Pivot={val_pivot:.4f}")

# C8 — Calibración: hit-rates en rango [0, 1]
calib_cols_all = [c for c in datos.columns if c.startswith("calib_q")]
if calib_cols_all:
    calib_min = datos[calib_cols_all].min().min()
    calib_max = datos[calib_cols_all].max().max()
    _chk("Hit-rates de calibración en [0, 1]",
         0.0 <= calib_min and calib_max <= 1.0,
         f"Observado: [{calib_min:.3f}, {calib_max:.3f}]")

# C9 — Monotonía RMSE: h02-05 ≤ h51-75 para cada config
if "rmse" in datos.columns:
    for cfg in cfgs_ok:
        df_ = dfs[cfg].copy()
        df_["h_grupo5"] = _h_grupo5(df_["h"])
        r_corto = df_[df_["h_grupo5"] == "h02-05"]["rmse"].mean()
        r_largo  = df_[df_["h_grupo5"] == "h51-75"]["rmse"].mean()
        _chk(f"{cfg}: RMSE h02-05 ≤ h51-75",
             r_corto <= r_largo,
             f"h02-05={r_corto/1e6:.0f} MM, h51-75={r_largo/1e6:.0f} MM")

# C10 — Ranking tiene winner
_chk("Ranking generado con ganador",
     winner is not None,
     f"Ganador: {winner}")

n_ok   = sum(1 for s, _, _ in checks if s == "✓")
n_fail = sum(1 for s, _, _ in checks if s == "✗")
print(f"\n  Resultado: {n_ok} ✓ / {n_fail} ✗ de {len(checks)} verificaciones")
print()

# Guardar checklist en Excel
chk_df = pd.DataFrame(checks, columns=["Estado", "Verificación", "Detalle"])
with pd.ExcelWriter(ruta_excel, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    chk_df.to_excel(writer, sheet_name="Checklist", index=False)
print(f"[OK] Checklist guardado en {ruta_excel}")


# ── 11. Reporte HTML ─────────────────────────────────────────────────────────
_COV_TARGETS = {
    "coverage_90"    : 0.90,
    "val_coverage_90": 0.90,
    "coverage_98"    : 0.98,
}

def _color_cell(val_raw, col, cfg, all_cfg_vals):
    """Devuelve clase CSS según si el valor es el mejor, peor o intermedio."""
    if pd.isna(val_raw):
        return ""
    vals = {c: v for c, v in all_cfg_vals.items() if not pd.isna(v)}
    if len(vals) < 2:
        return ""
    target = _COV_TARGETS.get(col)
    if target is not None:
        # mejor = más cercano al objetivo de cobertura
        best  = min(vals, key=lambda c: abs(vals[c] - target))
        worst = max(vals, key=lambda c: abs(vals[c] - target))
    else:
        # métricas de error: menor es mejor
        best  = min(vals, key=lambda c: vals[c])
        worst = max(vals, key=lambda c: vals[c])
    if cfg == best:
        return "best"
    if cfg == worst:
        return "worst"
    return ""


CSS = """
:root {
  --ink:#0F172A; --mid:#334155; --muted:#64748B; --border:#CBD5E1;
  --stripe:#F8FAFC; --bg:#FFFFFF; --teal:#0F766E; --teal-lt:#CCFBF1;
  --em:#059669; --em-lt:#D1FAE5; --red:#DC2626; --red-lt:#FEE2E2;
  --amber:#D97706; --amber-lt:#FEF3C7;
}
@media (prefers-color-scheme:dark){
  :root{--ink:#E2E8F0;--mid:#94A3B8;--muted:#64748B;--border:#1E293B;
    --stripe:#0F172A;--bg:#111827;--teal:#2DD4BF;--teal-lt:#134e4a;
    --em:#34D399;--em-lt:#064e3b;--red:#F87171;--red-lt:#450a0a;
    --amber:#FBBF24;--amber-lt:#451a03;}
}
:root[data-theme="light"]{--ink:#0F172A;--mid:#334155;--muted:#64748B;--border:#CBD5E1;--stripe:#F8FAFC;--bg:#FFFFFF;--teal:#0F766E;--teal-lt:#CCFBF1;--em:#059669;--em-lt:#D1FAE5;--red:#DC2626;--red-lt:#FEE2E2;--amber:#D97706;--amber-lt:#FEF3C7;}
:root[data-theme="dark"]{--ink:#E2E8F0;--mid:#94A3B8;--muted:#64748B;--border:#1E293B;--stripe:#0F172A;--bg:#111827;--teal:#2DD4BF;--teal-lt:#134e4a;--em:#34D399;--em-lt:#064e3b;--red:#F87171;--red-lt:#450a0a;--amber:#FBBF24;--amber-lt:#451a03;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.6;color:var(--ink);background:var(--bg);max-width:1100px;margin:0 auto;padding:40px 32px 80px;}
h1{font-size:22px;font-weight:700;color:var(--ink);margin-bottom:4px;}
.sub{font-size:13px;color:var(--muted);margin-bottom:32px;}
h2{font-size:15px;font-weight:700;color:var(--ink);margin:32px 0 8px;padding-bottom:6px;border-bottom:2px solid var(--teal);}
h3{font-size:13px;font-weight:700;color:var(--mid);margin:20px 0 6px;}
p{color:var(--mid);font-size:13px;margin-bottom:8px;line-height:1.6;}
.tbl-wrap{overflow-x:auto;margin-bottom:20px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;}
thead th{background:var(--teal);color:white;padding:9px 12px;text-align:left;font-weight:600;font-size:12px;letter-spacing:.03em;white-space:nowrap;}
tbody tr:nth-child(even){background:var(--stripe);}
tbody tr:hover{background:var(--teal-lt);}
td{padding:8px 12px;color:var(--mid);border-bottom:1px solid var(--border);}
td:first-child{font-weight:600;color:var(--ink);}
td.best{background:var(--em-lt);color:var(--em);font-weight:700;}
td.worst{background:var(--red-lt);color:var(--red);}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-bottom:16px;flex-wrap:wrap;}
.legend span{display:flex;align-items:center;gap:6px;}
.dot{width:12px;height:12px;border-radius:2px;display:inline-block;}
.dot-best{background:var(--em-lt);border:1px solid var(--em);}
.dot-worst{background:var(--red-lt);border:1px solid var(--red);}
.winner-box{background:var(--em-lt);border:2px solid var(--em);border-radius:6px;padding:14px 20px;margin-bottom:24px;}
.winner-box .wt{font-size:13px;font-weight:700;color:var(--em);margin-bottom:4px;}
.winner-box .wd{font-size:13px;color:var(--mid);}
.img-wrap{text-align:center;margin-bottom:20px;}
.img-wrap img{max-width:100%;border:1px solid var(--border);border-radius:6px;}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--border);font-size:11.5px;color:var(--muted);text-align:center;}
"""

def _build_html() -> str:
    from datetime import date
    today = date.today().strftime("%d %b %Y")

    sections = []

    # ── Hero ──
    winner_txt = f"<strong>{winner}</strong>" if winner else "—"
    sections.append(f"""
<h1>Comparativa cv4 DIRECT — 4 Configuraciones</h1>
<p class="sub">Expanding vs Rolling · Test 0.5 vs 1 año &nbsp;|&nbsp; Banco: {BANCO} &nbsp;|&nbsp; {today}</p>
<div class="winner-box">
  <div class="wt">★ Configuración recomendada para producción: {winner_txt}</div>
  <div class="wd">Basado en menor distancia a Cob90 = 90% + menor RMSE y CRPS normalizados entre configuraciones.</div>
</div>
<div class="legend">
  <span><span class="dot dot-best"></span> Mejor valor en esa fila</span>
  <span><span class="dot dot-worst"></span> Peor valor en esa fila</span>
</div>
""")

    # ── Tabla 1: Resumen global ──
    sections.append("<h2>1 — Resumen Global</h2>")
    sections.append("<p>Media aritmética sobre todos los horizontes h=2..75 y todos los folds disponibles. "
                    "Para coberturas: verde = más cercano al objetivo (90% / 98%), rojo = más alejado. "
                    "Para métricas de error: verde = menor valor.</p>")
    # build with raw vals for coloring
    raw_global = {}
    for col, _, label in [
        ("rmse","mm","RMSE (MM USD)"),("crps","mm","CRPS (MM USD)"),
        ("coverage_90","pct","Cobertura 90%"),("coverage_98","pct","Cobertura 98%"),
        ("val_coverage_90","pct","VAL Cob 90%"),("winkler_90","mm","Winkler 90% (MM USD)"),
        ("pinball_q50","mm","Pinball Q50 (MM USD)"),
    ]:
        if col not in datos.columns:
            continue
        row_d = {"Métrica": label}
        rv = {}
        for cfg in cfgs_ok:
            v = dfs[cfg][col].mean() if col in dfs[cfg].columns else np.nan
            rv[cfg] = v
            row_d[cfg] = _fmt_pct(v) if "pct" in _ else _fmt_mm(v)
        raw_global[label] = rv

    rows1 = []
    for col, kind, label in [
        ("rmse","mm","RMSE (MM USD)"),("crps","mm","CRPS (MM USD)"),
        ("coverage_90","pct","Cobertura 90%"),("coverage_98","pct","Cobertura 98%"),
        ("val_coverage_90","pct","VAL Cob 90%"),("winkler_90","mm","Winkler 90% (MM USD)"),
        ("pinball_q50","mm","Pinball Q50 (MM USD)"),
    ]:
        if col not in datos.columns:
            continue
        row_d = {"Métrica": label}
        for cfg in cfgs_ok:
            v = dfs[cfg][col].mean() if col in dfs[cfg].columns else np.nan
            row_d[cfg] = _fmt_pct(v) if kind == "pct" else _fmt_mm(v)
        rows1.append(row_d)

    # generate table with color
    if rows1:
        cols1 = list(rows1[0].keys())
        th1 = "".join(f"<th>{c}</th>" for c in cols1)
        body1 = ""
        for row_d in rows1:
            label = row_d["Métrica"]
            rv = raw_global.get(label, {})
            col_key = next((c for c, _, l in [
                ("rmse","mm","RMSE (MM USD)"),("crps","mm","CRPS (MM USD)"),
                ("coverage_90","pct","Cobertura 90%"),("coverage_98","pct","Cobertura 98%"),
                ("val_coverage_90","pct","VAL Cob 90%"),("winkler_90","mm","Winkler 90% (MM USD)"),
                ("pinball_q50","mm","Pinball Q50 (MM USD)")] if l == label), None)
            tds = f"<td>{label}</td>"
            for cfg in cfgs_ok:
                cls = _color_cell(rv.get(cfg, np.nan), col_key or "", cfg, rv)
                tds += f'<td class="{cls}">{row_d.get(cfg,"—")}</td>'
            body1 += f"<tr>{tds}</tr>"
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th1}</tr></thead><tbody>{body1}</tbody></table></div>')

    # ── Tabla 2: RMSE por grupo horizonte ──
    sections.append("<h2>2 — RMSE por Grupo de Horizonte (MM USD)</h2>")
    sections.append("<p>Media del RMSE dentro de cada grupo. Verde = menor error en esa fila. Rojo = mayor.</p>")
    if "rmse" in datos.columns:
        rmse_p = datos.groupby(["h_grupo","config"],observed=True)["rmse"].mean().unstack("config")
        rows2 = []
        for grp in rmse_p.index:
            row_d = {"Grupo": str(grp)}
            rv = {}
            for cfg in cfgs_ok:
                v = rmse_p.loc[grp, cfg] if cfg in rmse_p.columns else np.nan
                rv[cfg] = v
                row_d[cfg] = _fmt_mm(v)
            rv_norm = {c: v for c, v in rv.items() if not pd.isna(v)}
            best_c = min(rv_norm, key=rv_norm.get) if rv_norm else None
            worst_c = max(rv_norm, key=rv_norm.get) if rv_norm else None
            row_d["_best"] = best_c
            row_d["_worst"] = worst_c
            row_d["_rv"] = rv
            rows2.append(row_d)
        cols2 = ["Grupo"] + cfgs_ok
        th2 = "".join(f"<th>{c}</th>" for c in cols2)
        body2 = ""
        for row_d in rows2:
            best_c = row_d.pop("_best", None)
            worst_c = row_d.pop("_worst", None)
            row_d.pop("_rv", None)
            tds = f"<td>{row_d['Grupo']}</td>"
            for cfg in cfgs_ok:
                cls = "best" if cfg == best_c else ("worst" if cfg == worst_c else "")
                tds += f'<td class="{cls}">{row_d.get(cfg,"—")}</td>'
            body2 += f"<tr>{tds}</tr>"
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th2}</tr></thead><tbody>{body2}</tbody></table></div>')

    # ── Tabla 3: Cobertura 90% por grupo horizonte ──
    sections.append("<h2>3 — Cobertura 90% [Q05–Q95] por Grupo de Horizonte</h2>")
    sections.append("<p>Verde = más cercano al objetivo 90%. Rojo = más alejado.</p>")
    if "coverage_90" in datos.columns:
        cob_p = datos.groupby(["h_grupo","config"],observed=True)["coverage_90"].mean().unstack("config")
        rows3 = []
        for grp in cob_p.index:
            row_d = {"Grupo": str(grp), "Objetivo": "90.0%"}
            rv = {}
            for cfg in cfgs_ok:
                v = cob_p.loc[grp, cfg] if cfg in cob_p.columns else np.nan
                rv[cfg] = v
                row_d[cfg] = _fmt_pct(v)
            rv_norm = {c: v for c, v in rv.items() if not pd.isna(v)}
            best_c = min(rv_norm, key=lambda c: abs(rv_norm[c]-0.90)) if rv_norm else None
            worst_c = max(rv_norm, key=lambda c: abs(rv_norm[c]-0.90)) if rv_norm else None
            row_d["_best"] = best_c; row_d["_worst"] = worst_c
            rows3.append(row_d)
        cols3 = ["Grupo", "Objetivo"] + cfgs_ok
        th3 = "".join(f"<th>{c}</th>" for c in cols3)
        body3 = ""
        for row_d in rows3:
            best_c = row_d.pop("_best", None); worst_c = row_d.pop("_worst", None)
            tds = f"<td>{row_d['Grupo']}</td><td>{row_d['Objetivo']}</td>"
            for cfg in cfgs_ok:
                cls = "best" if cfg == best_c else ("worst" if cfg == worst_c else "")
                tds += f'<td class="{cls}">{row_d.get(cfg,"—")}</td>'
            body3 += f"<tr>{tds}</tr>"
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th3}</tr></thead><tbody>{body3}</tbody></table></div>')

    # ── Tabla 4: Cobertura por fold ──
    sections.append("<h2>4 — Cobertura 90% por Fold (Detección de Drift)</h2>")
    sections.append("<p>Fold 5 es el periodo más reciente. Un colapso sistemático en Fold 5 a través de todas las configs indica drift macroeconómico, no de ventana.</p>")
    if "coverage_90" in datos.columns and "fold" in datos.columns:
        cob_f = datos.groupby(["fold","config"])["coverage_90"].mean().unstack("config")
        rows4 = []
        for fold in sorted(cob_f.index):
            row_d = {"Fold": f"Fold {fold}", "Objetivo": "90.0%"}
            rv = {}
            for cfg in cfgs_ok:
                v = cob_f.loc[fold, cfg] if cfg in cob_f.columns else np.nan
                rv[cfg] = v
                row_d[cfg] = _fmt_pct(v)
            rv_norm = {c: v for c, v in rv.items() if not pd.isna(v)}
            best_c = min(rv_norm, key=lambda c: abs(rv_norm[c]-0.90)) if rv_norm else None
            worst_c = max(rv_norm, key=lambda c: abs(rv_norm[c]-0.90)) if rv_norm else None
            row_d["_best"] = best_c; row_d["_worst"] = worst_c
            rows4.append(row_d)
        cols4 = ["Fold", "Objetivo"] + cfgs_ok
        th4 = "".join(f"<th>{c}</th>" for c in cols4)
        body4 = ""
        for row_d in rows4:
            best_c = row_d.pop("_best", None); worst_c = row_d.pop("_worst", None)
            tds = f"<td>{row_d['Fold']}</td><td>{row_d['Objetivo']}</td>"
            for cfg in cfgs_ok:
                cls = "best" if cfg == best_c else ("worst" if cfg == worst_c else "")
                tds += f'<td class="{cls}">{row_d.get(cfg,"—")}</td>'
            body4 += f"<tr>{tds}</tr>"
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th4}</tr></thead><tbody>{body4}</tbody></table></div>')

    # ── Tabla 4b: fold × h_grupo5 por config ──
    sections.append("<h2>4b — Cobertura 90% por Fold × Grupo Horizonte (por configuración)</h2>")
    sections.append("<p>Verde = ≥ 89% (cerca del objetivo). Rojo = &lt; 80% (drift severo).</p>")
    H5_COLS = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]
    for cfg, piv in pivot_fold_hgrp.items():
        sections.append(f"<h3>{cfg}</h3>")
        th_b = "<th>Fold</th>" + "".join(f"<th>{c}</th>" for c in H5_COLS)
        body_b = ""
        for fold_lbl in piv.index:
            tds = f"<td>{fold_lbl}</td>"
            for col in H5_COLS:
                v = piv.loc[fold_lbl, col] if col in piv.columns else np.nan
                txt = _fmt_pct(v)
                cls = ""
                if not pd.isna(v):
                    if v >= 0.89:
                        cls = "best"
                    elif v < 0.80:
                        cls = "worst"
                tds += f'<td class="{cls}">{txt}</td>'
            body_b += f"<tr>{tds}</tr>"
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th_b}</tr></thead><tbody>{body_b}</tbody></table></div>')

    # ── Tabla 5: Calibración ──
    sections.append("<h2>5 — Calibración Hit-Rate por Cuantil</h2>")
    sections.append("<p>Hit-rate empírico vs τ nominal. Ideal: todas las celdas ≈ Nominal. Verde = más cercano, Rojo = más alejado.</p>")
    if calib_rows:
        cols5 = list(calib_rows[0].keys())
        th5 = "".join(f"<th>{c}</th>" for c in cols5)
        body5 = ""
        for row_d in calib_rows:
            tau_val = float(row_d["Nominal"].strip("%")) / 100
            rv = {}
            for cfg in cfgs_ok:
                try:
                    rv[cfg] = float(row_d.get(cfg,"0%").strip("%")) / 100
                except Exception:
                    rv[cfg] = np.nan
            rv_valid = {c: v for c, v in rv.items() if not pd.isna(v)}
            if not rv_valid:
                best_c = worst_c = None
            else:
                best_c  = min(rv_valid, key=lambda c: abs(rv_valid[c] - tau_val))
                worst_c = max(rv_valid, key=lambda c: abs(rv_valid[c] - tau_val))
            tds = ""
            for c in cols5:
                cls = ""
                if c in cfgs_ok:
                    cls = "best" if c == best_c else ("worst" if c == worst_c else "")
                tds += f'<td class="{cls}">{row_d.get(c,"—")}</td>'
            body5 += f"<tr>{tds}</tr>"
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th5}</tr></thead><tbody>{body5}</tbody></table></div>')

    # ── Tabla 6: Estadísticos descriptivos ──
    sections.append("<h2>6 — Estadísticos Descriptivos por Configuración</h2>")
    for col, label in [("rmse","RMSE (MM USD)"),("crps","CRPS (MM USD)"),
                        ("coverage_90","Cobertura 90%"),("winkler_90","Winkler 90% (MM USD)"),
                        ("pinball_q50","Pinball Q50 (MM USD)")]:
        if col not in datos.columns:
            continue
        is_pct = col in ("coverage_90","coverage_98")
        sections.append(f"<h3>{label}</h3>")
        tbl_s = datos.groupby("config")[col].agg(["mean","std","min","max"])
        rows6 = []
        for cfg in tbl_s.index:
            row_d = {"Config": cfg}
            for stat in ["mean","std","min","max"]:
                v = tbl_s.loc[cfg, stat]
                row_d[stat.capitalize()] = _fmt_pct(v) if is_pct else _fmt_mm(v)
            rows6.append(row_d)
        cols6 = ["Config","Mean","Std","Min","Max"]
        th6 = "".join(f"<th>{c}</th>" for c in cols6)
        body6 = "".join(f"<tr>{''.join(f'<td>{row_d.get(c,chr(8212))}</td>' for c in cols6)}</tr>"
                        for row_d in rows6)
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th6}</tr></thead><tbody>{body6}</tbody></table></div>')

    # ── Ranking ──
    sections.append("<h2>7 — Ranking Final</h2>")
    rank_rows = []
    for rank, (cfg, (score, parts)) in enumerate(sorted_configs, 1):
        rank_rows.append({
            "Posición": f"#{rank}",
            "Config": cfg,
            "Score compuesto": f"{score:.4f}",
            "Detalle": " | ".join(parts),
            "": "★ RECOMENDADO" if rank == 1 else ""
        })
    if rank_rows:
        cols_r = list(rank_rows[0].keys())
        th_r = "".join(f"<th>{c}</th>" for c in cols_r)
        body_r = "".join(
            f"<tr>{''.join(f'<td class={chr(34)}best{chr(34)}>{v}</td>' if i==0 and row_d.get(chr(35)+'1',row_d.get('Posición',''))=='#1' else f'<td>{v}</td>' for i,(k,v) in enumerate(row_d.items()))}</tr>"
            for row_d in rank_rows
        )
        sections.append(f'<div class="tbl-wrap"><table><thead><tr>{th_r}</tr></thead><tbody>{body_r}</tbody></table></div>')

    # ── Checklist HTML ──
    sections.append("<h2>8 — Checklist de Consistencia</h2>")
    sections.append("<p>Verificaciones automáticas sobre los datos cargados. Todas deben mostrar ✓ antes de interpretar los resultados.</p>")
    chk_th = "<th>Estado</th><th>Verificación</th><th>Detalle</th>"
    chk_body = ""
    for sym, label, detalle in checks:
        cls = "best" if sym == "✓" else "worst"
        chk_body += f'<tr><td class="{cls}" style="text-align:center;font-size:16px">{sym}</td><td>{label}</td><td style="font-size:12px;color:var(--muted)">{detalle}</td></tr>'
    sections.append(f'<div class="tbl-wrap"><table><thead><tr>{chk_th}</tr></thead><tbody>{chk_body}</tbody></table></div>')
    resumen_chk = f"<p><strong>{sum(1 for s,_,_ in checks if s=='✓')} ✓</strong> / <strong style='color:var(--red)'>{sum(1 for s,_,_ in checks if s=='✗')} ✗</strong> de {len(checks)} verificaciones</p>"
    sections.append(resumen_chk)

    sections.append("<footer>Sistema Inteligente cv4 DIRECT · Análisis Comparativo · Confidencial</footer>")

    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Comparativa cv4 — {BANCO}</title>
<style>{CSS}</style></head><body>
{"".join(sections)}
</body></html>"""


html_content = _build_html()
ruta_html = DIR_OUT / f"comparativo_configs_{BANCO}.html"
ruta_html.write_text(html_content, encoding="utf-8")
print(f"[OK] Reporte HTML: {ruta_html}")
print(f"\nAbre el archivo en tu navegador para ver las tablas comparativas.")

