# -*- coding: utf-8 -*-
"""
step006_cqr_calibration.py
==========================
Calibración post-hoc de cuantiles mediante Conformal Quantile Regression (Alt. 3)
+ Isotonic sort para garantizar monotonía Q01 < Q05 < ... < Q99.

Diseño:
  1. Por cada fold, lee preds_val_fold*.parquet  (predicciones en VAL)
  2. Calcula delta_tau por (h_grupo × tau):
       delta_tau = quantile_tau( Y_val - Q_tau_val )
  3. Aplica shift a preds_base (TEST): Q_tau_cal = Q_tau + delta_tau
  4. Aplica isotonic sort fila a fila para restaurar monotonía
  5. Guarda preds_calibrado_*.parquet con columnas idénticas a preds_base
  6. Genera tabla comparativa de métricas (antes / después) por h_grupo × tau
  7. Guarda calibration_params_*.json para uso en producción

Referencia teórica:
  Romano, Sesia & Candès (NeurIPS 2019) "Conformal Quantile Regression"
  Aplicada por cuantil individual (split conformal per-quantile shift).

Uso:
  - Cambiar _MODO_SUFIJO y BANCO según la configuración a calibrar
  - Ejecutar después de step005 (requiere preds_val_fold* en el directorio)
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"
DIR_CQR      = BASE_SISTEMA / "2. Output" / "step006_cqr"

_MODO_SUFIJO = "exp_arctan"   # "exp_0.5" | "exp_1" | "roll_0.5" | "roll_1"
                             # Agregar "_arctan" al sufijo para leer resultados de AJUSTE_ARCTAN=True
                             # Ejemplo: "roll_0.5_arctan"
BANCO        = "SISTEMA"

DIR_FOLD = DIR_WFCV / f"fold_{_MODO_SUFIJO}"
DIR_OUT  = DIR_CQR  / f"cqr_{_MODO_SUFIJO}"
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Cuantiles en el mismo orden que los modelos de step005
TAUS    = [0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99]
Q_COLS  = [f"q{int(t*100):02d}" for t in TAUS]   # ["q01","q05","q40","q50","q60","q95","q99"]

# Grupos de horizonte para calibración local
H_GRUPOS = {
    "muy_corto": list(range(2,  6)),    # h=2–5
    "corto"    : list(range(6,  21)),   # h=6–20
    "medio"    : list(range(21, 51)),   # h=21–50
    "largo"    : list(range(51, 76)),   # h=51–75
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _h_grupo_map(h: int) -> str:
    if h <= 5:  return "muy_corto"
    if h <= 20: return "corto"
    if h <= 50: return "medio"
    return "largo"


def _crps_approx(y: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """CRPS ≈ 2 × trapz pinball sobre los taus disponibles."""
    taus_sorted = sorted(t for t in preds if isinstance(t, float))
    pb = []
    for tau in taus_sorted:
        err = y - preds[tau]
        pb.append(float(np.where(err >= 0, tau * err, (tau - 1) * err).mean()))
    return float(2.0 * np.trapz(pb, taus_sorted))


def _metricas_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Calcula hit-rate por tau, cobertura 90%/98%, CRPS y sharpness
    agrupado por h_grupo. Devuelve DataFrame largo con columna 'version'.
    """
    df = df.copy()
    df["h_grupo"] = df["h"].apply(_h_grupo_map)
    rows = []

    for grupo in list(H_GRUPOS.keys()) + ["TOTAL"]:
        sub = df if grupo == "TOTAL" else df[df["h_grupo"] == grupo]
        if sub.empty:
            continue

        y = sub["target"].values
        row: dict = {"version": label, "h_grupo": grupo, "n": len(sub)}

        # Hit-rate por cuantil
        for tau, col in zip(TAUS, Q_COLS):
            if col in sub.columns:
                row[f"hitrate_q{int(tau*100):02d}"] = float((y <= sub[col].values).mean())

        # Cobertura de intervalo
        if "q05" in sub.columns and "q95" in sub.columns:
            row["coverage_90"] = float(
                ((y >= sub["q05"].values) & (y <= sub["q95"].values)).mean()
            )
        if "q01" in sub.columns and "q99" in sub.columns:
            row["coverage_98"] = float(
                ((y >= sub["q01"].values) & (y <= sub["q99"].values)).mean()
            )

        # Sharpness — anchura media del intervalo 90%
        if "q05" in sub.columns and "q95" in sub.columns:
            row["sharpness_90_mm"] = float((sub["q95"] - sub["q05"]).mean() / 1e6)

        # CRPS aproximado
        preds_dict = {tau: sub[col].values
                      for tau, col in zip(TAUS, Q_COLS)
                      if col in sub.columns}
        if preds_dict:
            row["crps_mm"] = _crps_approx(y, preds_dict) / 1e6

        rows.append(row)

    return pd.DataFrame(rows)


def _isotonic_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica isotonic regression fila a fila sobre las columnas Q_COLS
    para garantizar Q01 ≤ Q05 ≤ Q40 ≤ Q50 ≤ Q60 ≤ Q95 ≤ Q99.
    Columnas no cuantil (banco, fold, fecha_t, h, target, mean) se preservan.
    """
    cols_presentes = [c for c in Q_COLS if c in df.columns]
    if not cols_presentes:
        return df

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    taus_presentes = [TAUS[Q_COLS.index(c)] for c in cols_presentes]

    mat = df[cols_presentes].values.copy()   # (n_obs, n_tau)
    for i in range(len(mat)):
        mat[i] = iso.fit_transform(taus_presentes, mat[i])

    df = df.copy()
    df[cols_presentes] = mat
    return df


# ── 1. Cargar parquets VAL y TEST ──────────────────────────────────────────────
print(f"\n[INFO] Modo: fold_{_MODO_SUFIJO} | Banco: {BANCO}")
print(f"[INFO] Directorio: {DIR_FOLD}")

# VAL — todos los folds disponibles
val_paths = sorted(DIR_FOLD.glob(f"preds_val_fold*_{BANCO}_*.parquet"))
if not val_paths:
    raise FileNotFoundError(
        f"No se encontraron preds_val_fold* en {DIR_FOLD}.\n"
        "Asegúrate de haber ejecutado step005 con la versión que guarda VAL scaffolds."
    )
df_val = pd.concat([pd.read_parquet(p) for p in val_paths], ignore_index=True)
print(f"[OK] VAL cargado: {len(df_val):,} filas | folds: {sorted(df_val['fold'].unique())}")

# TEST — preds_base consolidado
test_paths = sorted(DIR_FOLD.glob(f"preds_base_{BANCO}_*.parquet"))
if not test_paths:
    raise FileNotFoundError(f"No se encontró preds_base_{BANCO}_*.parquet en {DIR_FOLD}")
df_test = pd.read_parquet(test_paths[-1])
print(f"[OK] TEST cargado: {len(df_test):,} filas | folds: {sorted(df_test['fold'].unique())}")

folds_disponibles = sorted(set(df_val["fold"].unique()) & set(df_test["fold"].unique()))
print(f"[INFO] Folds con VAL y TEST: {folds_disponibles}")


# ── 2. Calcular delta_tau por (fold × h_grupo × tau) ──────────────────────────
print("\n[PASO 2] Calculando delta_tau por fold × h_grupo × tau...")

# Agregar h_grupo al VAL
df_val["h_grupo"] = df_val["h"].apply(_h_grupo_map)

# Estructura: delta_params[fold][h_grupo][tau] = delta (float)
delta_params: dict[int, dict[str, dict[float, float]]] = {}

print(f"\n{'Fold':>5} {'h_grupo':>10}  " +
      "  ".join(f"Q{int(t*100):02d}" for t in TAUS))
print("-" * 80)

for fold in folds_disponibles:
    delta_params[fold] = {}
    val_fold = df_val[df_val["fold"] == fold]

    for grupo in H_GRUPOS:
        val_g = val_fold[val_fold["h_grupo"] == grupo]
        if val_g.empty:
            delta_params[fold][grupo] = {tau: 0.0 for tau in TAUS}
            continue

        y_val = val_g["target"].values
        deltas: dict[float, float] = {}

        for tau, col in zip(TAUS, Q_COLS):
            if col not in val_g.columns:
                deltas[tau] = 0.0
                continue
            # Residuo con signo: positivo = Y > Q_tau (underprediction)
            # delta = tau-cuantil empírico del residuo
            # Si modelo bien calibrado → delta ≈ 0
            # Si underpredicts Q05 (demasiados Y < Q05) → delta < 0 (empuja Q05 down)
            residuos = y_val - val_g[col].values
            n        = len(residuos)
            # Nivel de confianza finito-sample: ⌈(n+1)τ⌉/n (Romano et al. 2019)
            nivel    = min(np.ceil((n + 1) * tau) / n, 1.0)
            deltas[tau] = float(np.quantile(residuos, nivel))

        delta_params[fold][grupo] = deltas

        # Print fila de diagnóstico
        delta_str = "  ".join(
            f"{deltas[t]/1e6:+.1f}MM" for t in TAUS
        )
        print(f"{fold:>5} {grupo:>10}  {delta_str}")

print()


# ── 3. Aplicar calibración al TEST ────────────────────────────────────────────
print("[PASO 3] Aplicando calibración a TEST...")

df_test["h_grupo"] = df_test["h"].apply(_h_grupo_map)
df_cal = df_test.copy()

n_aplicados = 0
for fold in folds_disponibles:
    mask_fold = df_cal["fold"] == fold
    for grupo in H_GRUPOS:
        mask = mask_fold & (df_cal["h_grupo"] == grupo)
        if not mask.any():
            continue
        for tau, col in zip(TAUS, Q_COLS):
            if col not in df_cal.columns:
                continue
            delta = delta_params[fold][grupo].get(tau, 0.0)
            df_cal.loc[mask, col] = df_cal.loc[mask, col] + delta
        n_aplicados += mask.sum()

print(f"[OK] Shifts aplicados: {n_aplicados:,} celdas")


# ── 4. Isotonic sort ─────────────────────────────────────────────────────────
print("[PASO 4] Isotonic sort (garantía Q01 ≤ Q05 ≤ ... ≤ Q99)...")

# Contar cruces ANTES
cruces_antes = 0
q_pairs = list(zip(Q_COLS[:-1], Q_COLS[1:]))
for c1, c2 in q_pairs:
    if c1 in df_cal.columns and c2 in df_cal.columns:
        cruces_antes += int((df_cal[c1] > df_cal[c2]).sum())
print(f"  Cruces de cuantiles antes: {cruces_antes:,}")

df_cal = _isotonic_sort(df_cal)

cruces_despues = 0
for c1, c2 in q_pairs:
    if c1 in df_cal.columns and c2 in df_cal.columns:
        cruces_despues += int((df_cal[c1] > df_cal[c2]).sum())
print(f"  Cruces de cuantiles después: {cruces_despues:,}")


# ── 5. Guardar parquet calibrado ──────────────────────────────────────────────
fecha_hoy = date.today().strftime("%Y%m%d")

# Eliminar columna auxiliar h_grupo antes de guardar
df_cal_save = df_cal.drop(columns=["h_grupo"], errors="ignore")
df_test_save = df_test.drop(columns=["h_grupo"], errors="ignore")

ruta_cal = DIR_OUT / f"preds_calibrado_{BANCO}_{fecha_hoy}.parquet"
df_cal_save.to_parquet(ruta_cal, index=False)
print(f"\n[OK] Guardado: {ruta_cal.name}  ({len(df_cal_save):,} filas)")


# ── 6. Tabla comparativa de métricas ─────────────────────────────────────────
print("\n[PASO 6] Métricas comparativas ANTES vs DESPUÉS...")

met_antes  = _metricas_df(df_test, "ANTES")
met_despues = _metricas_df(df_cal,  "DESPUÉS")
df_met = pd.concat([met_antes, met_despues], ignore_index=True)

# ── Tabla 1: Hit-rate por cuantil y h_grupo ──────────────────────────────────
print("\n── Hit-rate por cuantil (objetivo = tau nominal) ──")
print(f"{'h_grupo':>12} {'version':>9}  " +
      "  ".join(f"Q{int(t*100):02d}" for t in TAUS))
print("-" * 90)

for grupo in list(H_GRUPOS.keys()) + ["TOTAL"]:
    for ver in ["ANTES", "DESPUÉS"]:
        fila = df_met[(df_met["h_grupo"] == grupo) & (df_met["version"] == ver)]
        if fila.empty:
            continue
        hr_str = "  ".join(
            f"{fila[f'hitrate_q{int(t*100):02d}'].values[0]:.1%}"
            if f"hitrate_q{int(t*100):02d}" in fila.columns else "  —  "
            for t in TAUS
        )
        print(f"{grupo:>12} {ver:>9}  {hr_str}")
    if grupo != "TOTAL":
        print()

# ── Tabla 2: Cobertura + CRPS + Sharpness ────────────────────────────────────
print("\n── Cobertura de intervalo y error ──")
print(f"{'h_grupo':>12} {'version':>9}  {'Cob90%':>7}  {'Cob98%':>7}  "
      f"{'CRPS MM':>9}  {'Sharp90 MM':>11}")
print("-" * 70)

for grupo in list(H_GRUPOS.keys()) + ["TOTAL"]:
    for ver in ["ANTES", "DESPUÉS"]:
        fila = df_met[(df_met["h_grupo"] == grupo) & (df_met["version"] == ver)]
        if fila.empty:
            continue
        cov90 = f"{fila['coverage_90'].values[0]:.1%}" if "coverage_90" in fila else "—"
        cov98 = f"{fila['coverage_98'].values[0]:.1%}" if "coverage_98" in fila else "—"
        crps  = f"{fila['crps_mm'].values[0]:.1f}" if "crps_mm" in fila else "—"
        sharp = f"{fila['sharpness_90_mm'].values[0]:.1f}" if "sharpness_90_mm" in fila else "—"
        print(f"{grupo:>12} {ver:>9}  {cov90:>7}  {cov98:>7}  {crps:>9}  {sharp:>11}")
    if grupo != "TOTAL":
        print()

# ── Tabla 3: Cobertura 90% por fold ──────────────────────────────────────────
print("\n── Cobertura 90% por fold (TOTAL) ──")
print(f"{'Fold':>6}  {'ANTES':>7}  {'DESPUÉS':>8}  {'Δpp':>6}  {'CRPS Antes':>11}  {'CRPS Dsp':>9}")
print("-" * 60)
for fold in sorted(folds_disponibles):
    mf_a = _metricas_df(df_test[df_test["fold"] == fold], "ANTES")
    mf_d = _metricas_df(df_cal[df_cal["fold"]   == fold], "DESPUÉS")
    tot_a_f = mf_a[mf_a["h_grupo"] == "TOTAL"]
    tot_d_f = mf_d[mf_d["h_grupo"] == "TOTAL"]
    if tot_a_f.empty or tot_d_f.empty:
        continue
    cov_a = tot_a_f["coverage_90"].values[0] if "coverage_90" in tot_a_f.columns else float("nan")
    cov_d = tot_d_f["coverage_90"].values[0] if "coverage_90" in tot_d_f.columns else float("nan")
    crps_a = tot_a_f["crps_mm"].values[0]    if "crps_mm"    in tot_a_f.columns else float("nan")
    crps_d = tot_d_f["crps_mm"].values[0]    if "crps_mm"    in tot_d_f.columns else float("nan")
    dpp = (cov_d - cov_a) * 100
    print(f"{fold:>6}  {cov_a:>7.1%}  {cov_d:>8.1%}  {dpp:>+6.1f}  "
          f"{crps_a:>11.1f}  {crps_d:>9.1f}")

# ── Tabla 4: Deltas (ajuste CQR) por fold × h_grupo × cuantil ────────────────
print("\n── Ajuste CQR (delta_tau) por fold × h_grupo × cuantil (MM USD) ──")
print(f"{'Fold':>5}  {'h_grupo':>10}  " + "  ".join(f"Q{int(t*100):02d}" for t in TAUS))
print("-" * 90)
for fold in sorted(folds_disponibles):
    for grupo in H_GRUPOS:
        deltas_g = delta_params[fold].get(grupo, {})
        if not deltas_g:
            continue
        vals = "  ".join(f"{deltas_g.get(t, 0.0)/1e6:+6.1f}" for t in TAUS)
        print(f"{fold:>5}  {grupo:>10}  {vals}")
    print()

# Guardar métricas en parquet
ruta_met = DIR_OUT / f"metricas_comparativo_{BANCO}_{fecha_hoy}.parquet"
df_met.to_parquet(ruta_met, index=False)
print(f"[OK] Métricas guardadas: {ruta_met.name}")


# ── 7. Guardar calibration params JSON (para producción) ─────────────────────
print("\n[PASO 7] Guardando parámetros de calibración para producción...")

# Para producción se usan los deltas del último fold (más reciente)
fold_prod = max(folds_disponibles)
deltas_prod = delta_params[fold_prod]

calib_json = {
    "banco"            : BANCO,
    "modo"             : _MODO_SUFIJO,
    "fecha_generacion" : fecha_hoy,
    "fold_produccion"  : int(fold_prod),
    "taus"             : [float(t) for t in TAUS],
    "q_cols"           : Q_COLS,
    "h_grupos"         : {g: [int(h) for h in hs] for g, hs in H_GRUPOS.items()},
    "calibration"      : {
        grupo: {
            f"q{int(tau*100):02d}": round(float(delta), 2)
            for tau, delta in deltas_grupo.items()
        }
        for grupo, deltas_grupo in deltas_prod.items()
    },
    "nota": (
        "Aplicar: Q_tau_cal = Q_tau_raw + calibration[h_grupo][q_col]. "
        "Después aplicar isotonic sort sobre [q01..q99] por fila. "
        "Basado en fold_produccion (fold más reciente disponible)."
    ),
}

ruta_json = DIR_OUT / f"calibration_params_{BANCO}_{fecha_hoy}.json"
with open(ruta_json, "w", encoding="utf-8") as fh:
    json.dump(calib_json, fh, indent=2, ensure_ascii=False)
print(f"[OK] Parámetros producción: {ruta_json.name}")

# Resumen de deltas de producción en consola
print(f"\nDeltas fold_prod={fold_prod} (MM USD):")
print(f"{'h_grupo':>12}  " + "  ".join(f"Q{int(t*100):02d}" for t in TAUS))
print("-" * 70)
for grupo, deltas_g in deltas_prod.items():
    vals = "  ".join(f"{deltas_g[t]/1e6:+.1f}" for t in TAUS)
    print(f"{grupo:>12}  {vals}")


# ── 8. Plots comparativos ─────────────────────────────────────────────────────
print("\n[PASO 8] Generando gráficos comparativos...")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle(
        f"CQR Calibración — {BANCO} · fold_{_MODO_SUFIJO}\n"
        f"Antes (azul) vs Después (verde) · {fecha_hoy}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    GRUPOS_ORDEN = list(H_GRUPOS.keys())
    COLOR_ANTES   = "#1E3A5F"
    COLOR_DESPUES = "#059669"
    TAU_LABELS    = [f"Q{int(t*100):02d}" for t in TAUS]

    # ── Panel 1: Hit-rate por tau (TOTAL) ────────────────────────────────────
    ax = axes[0, 0]
    hr_antes   = [met_antes[met_antes["h_grupo"]=="TOTAL"][f"hitrate_q{int(t*100):02d}"].values[0]
                  for t in TAUS if f"hitrate_q{int(t*100):02d}" in met_antes.columns]
    hr_despues = [met_despues[met_despues["h_grupo"]=="TOTAL"][f"hitrate_q{int(t*100):02d}"].values[0]
                  for t in TAUS if f"hitrate_q{int(t*100):02d}" in met_despues.columns]
    x = np.arange(len(TAUS))
    ax.plot(x, TAUS, "k--", lw=1.5, label="Nominal τ", zorder=5)
    ax.plot(x, hr_antes,   "o-", color=COLOR_ANTES,   lw=2, ms=7, label="Antes CQR")
    ax.plot(x, hr_despues, "s-", color=COLOR_DESPUES, lw=2, ms=7, label="Después CQR")
    ax.set_xticks(x); ax.set_xticklabels(TAU_LABELS, fontsize=9)
    ax.set_ylabel("Hit-rate empírico"); ax.set_title("Hit-rate por cuantil (TOTAL)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # ── Panel 2: Cobertura 90% por h_grupo ───────────────────────────────────
    ax = axes[0, 1]
    cov_a = [met_antes[met_antes["h_grupo"]==g]["coverage_90"].values[0]
             if "coverage_90" in met_antes.columns else np.nan for g in GRUPOS_ORDEN]
    cov_d = [met_despues[met_despues["h_grupo"]==g]["coverage_90"].values[0]
             if "coverage_90" in met_despues.columns else np.nan for g in GRUPOS_ORDEN]
    xi = np.arange(len(GRUPOS_ORDEN))
    bw = 0.32
    ax.bar(xi - bw/2, cov_a, width=bw, color=COLOR_ANTES,   alpha=0.85, label="Antes")
    ax.bar(xi + bw/2, cov_d, width=bw, color=COLOR_DESPUES, alpha=0.85, label="Después")
    ax.axhline(0.90, color="red", lw=1.5, ls="--", label="Objetivo 90%")
    ax.set_xticks(xi); ax.set_xticklabels(GRUPOS_ORDEN, fontsize=9)
    ax.set_ylabel("Cobertura empírica"); ax.set_title("Cobertura 90% por h_grupo", fontweight="bold")
    ax.set_ylim(0.6, 1.02); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for i, (a, d) in enumerate(zip(cov_a, cov_d)):
        ax.text(i - bw/2, a + 0.005, f"{a:.1%}", ha="center", va="bottom", fontsize=7, color=COLOR_ANTES)
        ax.text(i + bw/2, d + 0.005, f"{d:.1%}", ha="center", va="bottom", fontsize=7, color=COLOR_DESPUES)

    # ── Panel 3: CRPS por h_grupo ─────────────────────────────────────────────
    ax = axes[1, 0]
    crps_a = [met_antes[met_antes["h_grupo"]==g]["crps_mm"].values[0]
              if "crps_mm" in met_antes.columns else np.nan for g in GRUPOS_ORDEN]
    crps_d = [met_despues[met_despues["h_grupo"]==g]["crps_mm"].values[0]
              if "crps_mm" in met_despues.columns else np.nan for g in GRUPOS_ORDEN]
    ax.bar(xi - bw/2, crps_a, width=bw, color=COLOR_ANTES,   alpha=0.85, label="Antes")
    ax.bar(xi + bw/2, crps_d, width=bw, color=COLOR_DESPUES, alpha=0.85, label="Después")
    ax.set_xticks(xi); ax.set_xticklabels(GRUPOS_ORDEN, fontsize=9)
    ax.set_ylabel("CRPS (MM USD)"); ax.set_title("CRPS por h_grupo (menor = mejor)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 4: Deltas de calibración producción por h_grupo × tau ──────────
    ax = axes[1, 1]
    delta_mat = np.array([
        [deltas_prod[g][tau] / 1e6 for tau in TAUS]
        for g in GRUPOS_ORDEN
    ])   # (4 grupos, 7 taus)
    im = ax.imshow(delta_mat, aspect="auto", cmap="RdBu_r",
                   vmin=-max(abs(delta_mat.min()), abs(delta_mat.max())),
                   vmax= max(abs(delta_mat.min()), abs(delta_mat.max())))
    plt.colorbar(im, ax=ax, label="Delta (MM USD)")
    ax.set_xticks(range(len(TAUS))); ax.set_xticklabels(TAU_LABELS, fontsize=9)
    ax.set_yticks(range(len(GRUPOS_ORDEN))); ax.set_yticklabels(GRUPOS_ORDEN, fontsize=9)
    ax.set_title(f"Δ calibración fold={fold_prod} (MM USD)\nAzul=ajuste negativo · Rojo=ajuste positivo",
                 fontweight="bold", fontsize=10)
    for i in range(len(GRUPOS_ORDEN)):
        for j in range(len(TAUS)):
            ax.text(j, i, f"{delta_mat[i,j]:+.0f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(delta_mat[i,j]) > delta_mat.max()*0.5 else "black")

    ruta_fig = DIR_OUT / f"cqr_comparativo_{BANCO}_{fecha_hoy}.png"
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico: {ruta_fig.name}")

except Exception as e:
    print(f"[WARN] No se pudo generar gráfico: {e}")


# ── Resumen final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN CQR CALIBRACIÓN")
print("=" * 60)

# Extraer métricas TOTAL
tot_a = met_antes[met_antes["h_grupo"]  == "TOTAL"].iloc[0]
tot_d = met_despues[met_despues["h_grupo"] == "TOTAL"].iloc[0]

if "coverage_90" in tot_a.index:
    delta_cov = (tot_d["coverage_90"] - tot_a["coverage_90"]) * 100
    print(f"  Cobertura 90%:  {tot_a['coverage_90']:.1%} → {tot_d['coverage_90']:.1%}  "
          f"({delta_cov:+.1f}pp)")
if "crps_mm" in tot_a.index:
    delta_crps = tot_d["crps_mm"] - tot_a["crps_mm"]
    print(f"  CRPS (MM USD):  {tot_a['crps_mm']:.1f} → {tot_d['crps_mm']:.1f}  "
          f"({delta_crps:+.1f} MM)")
if "sharpness_90_mm" in tot_a.index:
    delta_sh = tot_d["sharpness_90_mm"] - tot_a["sharpness_90_mm"]
    print(f"  Sharpness 90%:  {tot_a['sharpness_90_mm']:.1f} → {tot_d['sharpness_90_mm']:.1f} MM  "
          f"({delta_sh:+.1f} MM — mayor anchura es el costo esperado)")

print(f"\n  Archivos generados en: {DIR_OUT}")
print(f"  → preds_calibrado_{BANCO}_{fecha_hoy}.parquet")
print(f"  → calibration_params_{BANCO}_{fecha_hoy}.json  ← usar en producción")
print(f"  → metricas_comparativo_{BANCO}_{fecha_hoy}.parquet")
print(f"  → cqr_comparativo_{BANCO}_{fecha_hoy}.png")
print("\n[DONE]")
