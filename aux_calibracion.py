# -*- coding: utf-8 -*-
"""
aux_calibracion.py
Diagnóstico de calibración del modelo cuantílico sobre el conjunto TEST.

Métricas implementadas:
  1. Calibration Plot   : cobertura empírica vs nominal para Q01…Q99
  2. Coverage by h      : % de realizados dentro de Q05-Q95 y Q01-Q99 para cada h
  3. CRPS by h          : Continuous Ranked Probability Score por horizonte
  4. Winkler Score      : cobertura + penalización por salidas del intervalo
  5. CQR (opcional)     : Conformalized Quantile Regression — garantía finita usando VAL

Lógica:
  - Se predicen TODAS las (fecha_t, h) del TEST que tienen target realizado.
  - La mediana de imputación se calcula sobre TRAIN (sin leak).
  - CQR usa el VAL como conjunto de calibración.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_calibracion"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO      = "SISTEMA"
CORTE_VAL  = pd.Timestamp("2022-07-01")   # alineado con step003
CORTE_TEST = pd.Timestamp("2023-01-03")   # inicio de datos de tasas (allocation)

ALPHA_CQR  = 0.10   # cobertura objetivo CQR = 1 - ALPHA_CQR = 90 %


# ── 1. Cargar modelo ──────────────────────────────────────────────────────────
def cargar_modelos(banco, dir_modelos):
    metas = sorted(dir_modelos.glob(f"metadata_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(f"No metadata para banco={banco}")
    meta  = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / f"lgbm_{banco}_q{int(tau*100):02d}_{fecha}.txt"
        if not ruta.exists():
            raise FileNotFoundError(ruta)
        modelos[tau] = lgb.Booster(model_file=str(ruta))

    print(f"Modelo: {metas[0].name}  |  quantiles={quantiles}  |  features={len(cols_feat)}")
    return modelos, cols_feat, quantiles


# ── 2. Datos + split ──────────────────────────────────────────────────────────
def cargar_splits(banco, cols_feat):
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    cols_excluir = {"fecha_t", "banco", "target"}
    cols_num = [c for c in df.columns if c not in cols_excluir]
    medianas = df[df["fecha_t"] < CORTE_VAL][cols_num].median()

    def imputar(df_):
        d = df_.copy()
        ok = [c for c in cols_num if c in d.columns]
        d[ok] = d[ok].fillna(medianas)
        for c in set(cols_feat) - set(d.columns):
            d[c] = 0.0
        return d

    df_val  = imputar(df[(df["fecha_t"] >= CORTE_VAL)  & (df["fecha_t"] < CORTE_TEST)])
    df_test = imputar(df[df["fecha_t"] >= CORTE_TEST])

    print(f"TRAIN: hasta {CORTE_VAL.date()}")
    print(f"VAL  : {CORTE_VAL.date()} → {df_val['fecha_t'].max().date()}"
          f"  ({df_val['fecha_t'].nunique()} fechas)")
    print(f"TEST : {CORTE_TEST.date()} → {df_test['fecha_t'].max().date()}"
          f"  ({df_test['fecha_t'].nunique()} fechas)")
    return df_val, df_test, cols_feat


# ── 3. Predecir un dataframe completo ────────────────────────────────────────
def predecir_df(modelos, df, cols_feat):
    """Devuelve df con columnas q01, q05, q50, q95, q99 + target."""
    cols_ok = [c for c in cols_feat if c in df.columns]
    X = df[cols_ok].copy()
    res = df[["fecha_t", "h", "target"]].copy().reset_index(drop=True)
    for tau, m in modelos.items():
        res[f"q{int(tau*100):02d}"] = m.predict(X)
    q_cols = sorted([c for c in res.columns if c.startswith("q")])
    res[q_cols] = np.sort(res[q_cols].values, axis=1)
    return res


# ── 4. Calibration Plot ───────────────────────────────────────────────────────
def calibration_plot(res_test, quantiles, banco):
    """
    Para cada τ en quantiles, calcula la fracción de realizados ≤ Q_τ.
    Debería ser ≈ τ (diagonal perfecta).
    """
    y = res_test["target"].dropna()
    idx = res_test["target"].notna()

    empiricos = []
    for tau in quantiles:
        col = f"q{int(tau*100):02d}"
        frac = (res_test.loc[idx, "target"].values <= res_test.loc[idx, col].values).mean()
        empiricos.append(frac)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(quantiles, empiricos, "o-", lw=2, ms=6, color="steelblue", label="Empírico")
    ax.plot([0, 1], [0, 1], "r--", lw=2, label="Calibración perfecta")
    ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0)

    # Banda de tolerancia ±5%
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                    alpha=0.10, color="green", label="Tolerancia ±5%")

    ax.set_xlabel("Cuantil nominal (τ)", fontsize=12)
    ax.set_ylabel("Fracción empírica ≤ Q_τ", fontsize=12)
    ax.set_title(f"Calibration Plot — {banco} (TEST)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / f"calibration_plot_{banco}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: calibration_plot_{banco}.png")

    # Resumen numérico
    print("\n  τ_nominal  τ_empírico  Δ")
    for tau, emp in zip(quantiles, empiricos):
        delta = emp - tau
        flag = "  ←" if abs(delta) > 0.05 else ""
        print(f"  {tau:.2f}       {emp:.3f}       {delta:+.3f}{flag}")


# ── 5. Coverage by h ──────────────────────────────────────────────────────────
def coverage_by_h(res_test, banco):
    """
    Para cada h=1..90:
      - % de realizados dentro de Q05-Q95  (objetivo: 90%)
      - % de realizados dentro de Q01-Q99  (objetivo: 98%)
    """
    idx = res_test["target"].notna()
    df  = res_test[idx].copy()

    hs   = sorted(df["h"].unique())
    cov_90 = []
    cov_98 = []
    n_obs  = []

    for h in hs:
        sub = df[df["h"] == h]
        n   = len(sub)
        if n == 0:
            cov_90.append(np.nan); cov_98.append(np.nan); n_obs.append(0)
            continue
        y  = sub["target"].values
        c90 = ((y >= sub["q05"].values) & (y <= sub["q95"].values)).mean()
        c98 = ((y >= sub["q01"].values) & (y <= sub["q99"].values)).mean()
        cov_90.append(c90); cov_98.append(c98); n_obs.append(n)

    hs_arr    = np.array(hs)
    cov_90_arr = np.array(cov_90)
    cov_98_arr = np.array(cov_98)

    # Suavizado con ventana móvil de 5 días
    def smooth(x, w=5):
        return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axhline(0.90, color="steelblue", lw=1.5, ls="--", alpha=0.7, label="Objetivo Q05-Q95 (90%)")
    ax.axhline(0.98, color="navy",      lw=1.5, ls="--", alpha=0.7, label="Objetivo Q01-Q99 (98%)")
    ax.plot(hs_arr, smooth(cov_90_arr), color="steelblue", lw=2,   label="Cobertura Q05-Q95 (suav.)")
    ax.plot(hs_arr, smooth(cov_98_arr), color="navy",      lw=2,   label="Cobertura Q01-Q99 (suav.)")
    ax.scatter(hs_arr, cov_90_arr, color="steelblue", s=14, alpha=0.4)
    ax.scatter(hs_arr, cov_98_arr, color="navy",      s=14, alpha=0.4)
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
    ax.set_ylabel("Cobertura empírica", fontsize=11)
    ax.set_title(f"Coverage by Horizon h — {banco} (TEST)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / f"coverage_by_h_{banco}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: coverage_by_h_{banco}.png")

    # Promedios globales
    valid = ~np.isnan(cov_90_arr)
    print(f"\nCobertura media Q05-Q95 : {np.nanmean(cov_90_arr):.1%}  (objetivo 90%)")
    print(f"Cobertura media Q01-Q99 : {np.nanmean(cov_98_arr):.1%}  (objetivo 98%)")
    print(f"Observaciones en TEST   : {sum(n_obs)} pares (fecha_t, h) con target realizado")
    return hs_arr, cov_90_arr, cov_98_arr


# ── 6. CRPS by h ─────────────────────────────────────────────────────────────
def crps_by_h(res_test, quantiles, banco):
    """
    CRPS = promedio de pinball loss sobre todos los cuantiles disponibles.
    Aproxima integral ∫ρ_τ dτ — menor = mejor.
    """
    idx = res_test["target"].notna()
    df  = res_test[idx].copy()

    hs     = sorted(df["h"].unique())
    crps_h = []

    q_cols = [f"q{int(tau*100):02d}" for tau in quantiles]

    for h in hs:
        sub = df[df["h"] == h]
        y   = sub["target"].values
        scores = []
        for tau, col in zip(quantiles, q_cols):
            r = y - sub[col].values
            pb = np.where(r >= 0, tau * r, (tau - 1) * r)
            scores.append(pb)
        crps_h.append(np.mean(scores))

    crps_h = np.array(crps_h) / 1e6   # en MM USD

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(hs, crps_h, color="steelblue", alpha=0.6, width=0.8)
    ax.plot(hs, pd.Series(crps_h).rolling(5, center=True, min_periods=1).mean(),
            color="crimson", lw=2, label="Suavizado 5 días")
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
    ax.set_ylabel("CRPS (MM USD)", fontsize=11)
    ax.set_title(f"CRPS por Horizonte h — {banco} (TEST)\n"
                 "Menor = mejor calibración distribucional", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / f"crps_by_h_{banco}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: crps_by_h_{banco}.png")
    print(f"\nCRPS medio (TEST): {crps_h.mean():.2f} MM USD")
    print(f"CRPS h=1-10      : {crps_h[:10].mean():.2f} MM USD  (corto plazo)")
    print(f"CRPS h=61-90     : {crps_h[60:].mean():.2f} MM USD  (largo plazo)")
    return np.array(hs), crps_h


# ── 7. Winkler Score ──────────────────────────────────────────────────────────
def winkler_by_h(res_test, banco, alpha=0.10):
    """
    Winkler Score para intervalo Q05-Q95 (90%):
      W = ancho + (2/alpha) * max(0, lower - y) + (2/alpha) * max(0, y - upper)
    Menor = mejor (combina sharpness + penalización por salidas).
    """
    idx = res_test["target"].notna()
    df  = res_test[idx].copy()
    hs  = sorted(df["h"].unique())
    wink = []

    for h in hs:
        sub   = df[df["h"] == h]
        y     = sub["target"].values / 1e6
        lower = sub["q05"].values / 1e6
        upper = sub["q95"].values / 1e6
        w = (upper - lower
             + (2 / alpha) * np.maximum(0, lower - y)
             + (2 / alpha) * np.maximum(0, y - upper))
        wink.append(w.mean())

    wink = np.array(wink)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(hs, wink, color="darkorange", alpha=0.6, width=0.8)
    ax.plot(hs, pd.Series(wink).rolling(5, center=True, min_periods=1).mean(),
            color="darkred", lw=2, label="Suavizado 5 días")
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
    ax.set_ylabel("Winkler Score (MM USD)", fontsize=11)
    ax.set_title(f"Winkler Score por Horizonte h — {banco} (TEST)\n"
                 "Intervalo Q05-Q95 (90%)  |  Menor = mejor", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / f"winkler_by_h_{banco}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: winkler_by_h_{banco}.png")
    print(f"\nWinkler medio (TEST): {wink.mean():.2f} MM USD")
    return np.array(hs), wink


# ── 8. CQR — Conformalized Quantile Regression ───────────────────────────────
def cqr_ajuste(res_val, res_test, banco, alpha=ALPHA_CQR):
    """
    CQR (Angelopoulos & Bates, 2021):
      1. Nonconformity score en VAL: s_i = max(Q05_i - y_i, y_i - Q95_i)
      2. Ajuste q̂ = cuantil ceil((1-α)(1+1/n)) de {s_i}
      3. Intervalo ajustado en TEST: [Q05 - q̂, Q95 + q̂]

    Garantía: P(y_test ∈ [Q05-q̂, Q95+q̂]) ≥ 1-α  (finita, sin suposición distribucional)
    """
    # ── Paso 1: scores de no-conformidad en VAL ──────────────────────────
    val_ok = res_val[res_val["target"].notna()].copy()
    if len(val_ok) == 0:
        print("  No hay realizados en VAL — CQR omitido.")
        return None

    s = np.maximum(
        val_ok["q05"].values - val_ok["target"].values,
        val_ok["target"].values - val_ok["q95"].values
    )

    # ── Paso 2: ajuste q̂ ────────────────────────────────────────────────
    n_cal = len(s)
    nivel = np.ceil((1 - alpha) * (1 + 1 / n_cal)) / (1 + 1 / n_cal)
    # Cuantil empírico corregido
    q_hat = np.quantile(s, min(nivel, 1.0))
    print(f"\nCQR — VAL calibración:")
    print(f"  n_cal = {n_cal}  |  q̂ = {q_hat/1e6:.1f} MM USD  |  "
          f"ajuste = {q_hat/1e6:.1f} MM USD por cada lado del intervalo")

    # ── Paso 3: cobertura con y sin ajuste en TEST ───────────────────────
    test_ok = res_test[res_test["target"].notna()].copy()
    y = test_ok["target"].values

    cov_raw = ((y >= test_ok["q05"].values) & (y <= test_ok["q95"].values)).mean()
    cov_cqr = ((y >= test_ok["q05"].values - q_hat) &
               (y <= test_ok["q95"].values + q_hat)).mean()

    print(f"\n  Cobertura Q05-Q95 sin CQR : {cov_raw:.1%}  (objetivo 90%)")
    print(f"  Cobertura Q05-Q95 con CQR : {cov_cqr:.1%}  (objetivo 90%, garantizado ≥ 90%)")

    # ── Ancho de intervalo medio ─────────────────────────────────────────
    ancho_raw = ((test_ok["q95"] - test_ok["q05"]).mean() / 1e6)
    ancho_cqr = ancho_raw + 2 * q_hat / 1e6
    print(f"\n  Ancho medio sin CQR : {ancho_raw:.1f} MM USD")
    print(f"  Ancho medio con CQR : {ancho_cqr:.1f} MM USD  "
          f"(costo de la garantía: +{2*q_hat/1e6:.1f} MM USD por lado)")

    # ── Coverage by h — comparativa ─────────────────────────────────────
    hs = sorted(test_ok["h"].unique())
    cov_raw_h, cov_cqr_h = [], []
    for h in hs:
        sub = test_ok[test_ok["h"] == h]
        yh  = sub["target"].values
        cov_raw_h.append(((yh >= sub["q05"].values) & (yh <= sub["q95"].values)).mean())
        cov_cqr_h.append(((yh >= sub["q05"].values - q_hat) &
                           (yh <= sub["q95"].values + q_hat)).mean())

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axhline(1 - alpha, color="green", lw=1.5, ls="--", alpha=0.8,
               label=f"Objetivo {1-alpha:.0%}")
    ax.plot(hs, pd.Series(cov_raw_h).rolling(5, center=True, min_periods=1).mean(),
            color="steelblue", lw=2, label="Sin CQR (Q05-Q95 raw)")
    ax.plot(hs, pd.Series(cov_cqr_h).rolling(5, center=True, min_periods=1).mean(),
            color="crimson", lw=2, label=f"Con CQR (garantía ≥ {1-alpha:.0%})")
    ax.scatter(hs, cov_raw_h, color="steelblue", s=12, alpha=0.35)
    ax.scatter(hs, cov_cqr_h, color="crimson",   s=12, alpha=0.35)
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
    ax.set_ylabel("Cobertura empírica", fontsize=11)
    ax.set_title(f"CQR vs Raw — Cobertura por h — {banco}\n"
                 f"Ajuste q̂ = {q_hat/1e6:.1f} MM USD por lado", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / f"cqr_coverage_by_h_{banco}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: cqr_coverage_by_h_{banco}.png")
    return q_hat


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modelos, cols_feat, quantiles = cargar_modelos(BANCO, DIR_MODELOS)
    df_val, df_test, cols_feat    = cargar_splits(BANCO, cols_feat)

    print("\nPrediciendo VAL y TEST...")
    res_val  = predecir_df(modelos, df_val,  cols_feat)
    res_test = predecir_df(modelos, df_test, cols_feat)

    n_test_real = res_test["target"].notna().sum()
    print(f"\nPares (fecha_t, h) con target realizado en TEST: {n_test_real:,}")

    print("\n" + "="*60)
    print("1. CALIBRATION PLOT")
    print("="*60)
    calibration_plot(res_test, quantiles, BANCO)

    print("\n" + "="*60)
    print("2. COVERAGE BY HORIZON h")
    print("="*60)
    coverage_by_h(res_test, BANCO)

    print("\n" + "="*60)
    print("3. CRPS BY HORIZON h")
    print("="*60)
    crps_by_h(res_test, quantiles, BANCO)

    print("\n" + "="*60)
    print("4. WINKLER SCORE BY HORIZON h")
    print("="*60)
    winkler_by_h(res_test, BANCO)

    print("\n" + "="*60)
    print("5. CQR — GARANTÍA DE COBERTURA FINITA")
    print("="*60)
    q_hat = cqr_ajuste(res_val, res_test, BANCO)

    print("\n✓ Diagnóstico completo. Outputs en:", DIR_OUTPUT)
