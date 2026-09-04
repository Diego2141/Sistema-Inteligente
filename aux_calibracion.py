# -*- coding: utf-8 -*-
"""
aux_calibracion.py
Diagnóstico de calibración comparativo: LightGBM vs XGBoost.

Estructura:
  PARTE 1 — LightGBM : carga modelos .txt, calcula 5 métricas, guarda PNGs _lgbm
  PARTE 2 — XGBoost  : carga modelos .json, calcula 5 métricas, guarda PNGs _xgb
  PARTE 3 — Comparación: plots superpuestos + tabla resumen lado a lado

Métricas:
  1. Calibration Plot   : cobertura empírica vs nominal Q01…Q99
  2. Coverage by h      : % dentro de Q05-Q95 y Q01-Q99 por horizonte
  3. CRPS by h          : Continuous Ranked Probability Score por horizonte
  4. Winkler Score by h : sharpness + penalización por salidas del intervalo
  5. CQR                : ajuste de garantía finita usando VAL como calibración
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_SISTEMA  = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ   = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MOD_LGB   = BASE_SISTEMA / "2. Output" / "modelos"     / "eval"
DIR_MOD_XGB   = BASE_SISTEMA / "2. Output" / "modelos_xgb" / "eval"
DIR_OUTPUT    = BASE_SISTEMA / "2. Output" / "aux_calibracion"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO      = "SISTEMA"
CORTE_VAL  = pd.Timestamp("2022-07-01")
CORTE_TEST = pd.Timestamp("2023-01-03")
CORTE_TEST_FIN = None  # None = toda la muestra TEST | pd.Timestamp("2023-12-31") = primer año
ALPHA_CQR  = 0.10   # cobertura objetivo = 1 - ALPHA_CQR = 90%

# Colores por modelo (consistentes con step003/step004 y fancharts)
ESTILO = {
    "lgbm": {"color": "steelblue",  "label": "LightGBM"},
    "xgb" : {"color": "darkorange", "label": "XGBoost"},
}


###############################################################################
# CARGA DE MODELOS
###############################################################################

def cargar_modelos_lgbm(banco: str, dir_modelos: Path):
    metas = sorted(dir_modelos.glob(f"metadata_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(f"No metadata LightGBM para banco={banco}")
    meta      = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha     = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / f"lgbm_{banco}_q{int(tau*100):02d}_{fecha}.txt"
        if not ruta.exists():
            raise FileNotFoundError(ruta)
        modelos[tau] = lgb.Booster(model_file=str(ruta))

    print(f"[LightGBM] {metas[0].name}  |  quantiles={quantiles}  |  features={len(cols_feat)}")
    return modelos, cols_feat, quantiles


def cargar_modelos_xgb(banco: str, dir_modelos: Path):
    metas = sorted(dir_modelos.glob(f"metadata_xgb_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(f"No metadata XGBoost para banco={banco}")
    meta      = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha     = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / f"xgb_{banco}_q{int(tau*100):02d}_{fecha}.json"
        if not ruta.exists():
            raise FileNotFoundError(ruta)
        booster = xgb.Booster()
        booster.load_model(str(ruta))
        modelos[tau] = booster

    print(f"[XGBoost]  {metas[0].name}  |  quantiles={quantiles}  |  features={len(cols_feat)}")
    return modelos, cols_feat, quantiles


###############################################################################
# DATOS + SPLIT (compartido)
###############################################################################

def cargar_splits(banco: str, cols_feat: list[str]):
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    cols_excluir = {"fecha_t", "banco", "target"}
    cols_num     = [c for c in df.columns if c not in cols_excluir]
    medianas     = df[df["fecha_t"] < CORTE_VAL][cols_num].median()

    def imputar(df_):
        d  = df_.copy()
        ok = [c for c in cols_num if c in d.columns]
        d[ok] = d[ok].fillna(medianas)
        for c in set(cols_feat) - set(d.columns):
            d[c] = 0.0
        return d

    df_val  = imputar(df[(df["fecha_t"] >= CORTE_VAL) & (df["fecha_t"] < CORTE_TEST)])
    mask_test = df["fecha_t"] >= CORTE_TEST
    if CORTE_TEST_FIN is not None:
        mask_test &= df["fecha_t"] <= CORTE_TEST_FIN
    df_test = imputar(df[mask_test])

    print(f"  VAL : {CORTE_VAL.date()} → {df_val['fecha_t'].max().date()}"
          f"  ({df_val['fecha_t'].nunique()} fechas)")
    fin_test = CORTE_TEST_FIN.date() if CORTE_TEST_FIN else df_test['fecha_t'].max().date()
    print(f"  TEST: {CORTE_TEST.date()} → {fin_test}"
          f"  ({df_test['fecha_t'].nunique()} fechas)"
          f"{'  [primer año]' if CORTE_TEST_FIN else ''}")
    return df_val, df_test


###############################################################################
# PREDICCIÓN
###############################################################################

def predecir_df_lgbm(modelos: dict, df: pd.DataFrame, cols_feat: list[str]) -> pd.DataFrame:
    cols_ok = [c for c in cols_feat if c in df.columns]
    X   = df[cols_ok].copy()
    res = df[["fecha_t", "h", "target"]].copy().reset_index(drop=True)
    for tau, m in modelos.items():
        res[f"q{int(tau*100):02d}"] = m.predict(X)
    q_cols = sorted([c for c in res.columns if c.startswith("q")])
    res[q_cols] = np.sort(res[q_cols].values, axis=1)
    return res


def predecir_df_xgb(modelos: dict, df: pd.DataFrame, cols_feat: list[str]) -> pd.DataFrame:
    cols_ok = [c for c in cols_feat if c in df.columns]
    dmat = xgb.DMatrix(df[cols_ok].copy())
    res  = df[["fecha_t", "h", "target"]].copy().reset_index(drop=True)
    for tau, m in modelos.items():
        res[f"q{int(tau*100):02d}"] = m.predict(dmat)
    q_cols = sorted([c for c in res.columns if c.startswith("q")])
    res[q_cols] = np.sort(res[q_cols].values, axis=1)
    return res


###############################################################################
# FUNCIONES DE MÉTRICAS (genéricas — devuelven valores para comparación)
###############################################################################

def calibration_plot(res_test: pd.DataFrame, quantiles: list, banco: str,
                     modelo_id: str, guardar: bool = True):
    idx       = res_test["target"].notna()
    empiricos = []
    for tau in quantiles:
        col  = f"q{int(tau*100):02d}"
        frac = (res_test.loc[idx, "target"].values <= res_test.loc[idx, col].values).mean()
        empiricos.append(frac)

    color = ESTILO[modelo_id]["color"]
    label = ESTILO[modelo_id]["label"]

    if guardar:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(quantiles, empiricos, "o-", lw=2, ms=6, color=color, label=f"Empírico ({label})")
        ax.plot([0, 1], [0, 1], "r--", lw=2, label="Calibración perfecta")
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                        alpha=0.10, color="green", label="Tolerancia ±5%")
        ax.set_xlabel("Cuantil nominal (τ)", fontsize=12)
        ax.set_ylabel("Fracción empírica ≤ Q_τ", fontsize=12)
        ax.set_title(f"Calibration Plot — {banco} [{label}] (TEST)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"calibration_plot_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print(f"\n  τ_nominal  τ_empírico  Δ  [{label}]")
    for tau, emp in zip(quantiles, empiricos):
        delta = emp - tau
        flag = "  ←" if abs(delta) > 0.05 else ""
        print(f"  {tau:.2f}       {emp:.3f}       {delta:+.3f}{flag}")

    return empiricos


def coverage_by_h(res_test: pd.DataFrame, banco: str, modelo_id: str,
                  guardar: bool = True):
    idx = res_test["target"].notna()
    df  = res_test[idx].copy()
    hs  = sorted(df["h"].unique())
    cov_90, cov_98 = [], []

    for h in hs:
        sub = df[df["h"] == h]
        y   = sub["target"].values
        cov_90.append(((y >= sub["q05"].values) & (y <= sub["q95"].values)).mean())
        cov_98.append(((y >= sub["q01"].values) & (y <= sub["q99"].values)).mean())

    hs_arr    = np.array(hs)
    cov_90_arr = np.array(cov_90)
    cov_98_arr = np.array(cov_98)
    color = ESTILO[modelo_id]["color"]
    label = ESTILO[modelo_id]["label"]

    def smooth(x, w=5):
        return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values

    if guardar:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.axhline(0.90, color=color, lw=1.5, ls="--", alpha=0.7,
                   label="Objetivo Q05-Q95 (90%)")
        ax.axhline(0.98, color="navy", lw=1.5, ls="--", alpha=0.7,
                   label="Objetivo Q01-Q99 (98%)")
        ax.plot(hs_arr, smooth(cov_90_arr), color=color, lw=2,
                label=f"Cob. Q05-Q95 ({label})")
        ax.plot(hs_arr, smooth(cov_98_arr), color="navy", lw=2,
                label=f"Cob. Q01-Q99 ({label})")
        ax.scatter(hs_arr, cov_90_arr, color=color, s=14, alpha=0.4)
        ax.scatter(hs_arr, cov_98_arr, color="navy", s=14, alpha=0.4)
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
        ax.set_ylabel("Cobertura empírica", fontsize=11)
        ax.set_title(f"Coverage by h — {banco} [{label}] (TEST)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"coverage_by_h_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print(f"\nCobertura media Q05-Q95 [{label}]: {np.nanmean(cov_90_arr):.1%}  (obj 90%)")
    print(f"Cobertura media Q01-Q99 [{label}]: {np.nanmean(cov_98_arr):.1%}  (obj 98%)")
    return hs_arr, cov_90_arr, cov_98_arr


def crps_by_h(res_test: pd.DataFrame, quantiles: list, banco: str,
              modelo_id: str, guardar: bool = True):
    idx    = res_test["target"].notna()
    df     = res_test[idx].copy()
    hs     = sorted(df["h"].unique())
    q_cols = [f"q{int(tau*100):02d}" for tau in quantiles]
    crps_h = []

    for h in hs:
        sub    = df[df["h"] == h]
        y      = sub["target"].values
        scores = []
        for tau, col in zip(quantiles, q_cols):
            r = y - sub[col].values
            scores.append(np.where(r >= 0, tau * r, (tau - 1) * r))
        crps_h.append(np.mean(scores))

    crps_h = np.array(crps_h) / 1e6
    color  = ESTILO[modelo_id]["color"]
    label  = ESTILO[modelo_id]["label"]

    if guardar:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(hs, crps_h, color=color, alpha=0.5, width=0.8)
        ax.plot(hs, pd.Series(crps_h).rolling(5, center=True, min_periods=1).mean(),
                color="crimson", lw=2, label="Suavizado 5 días")
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
        ax.set_ylabel("CRPS (MM USD)", fontsize=11)
        ax.set_title(f"CRPS por h — {banco} [{label}] (TEST)\nMenor = mejor",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"crps_by_h_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print(f"\nCRPS medio     [{label}]: {crps_h.mean():.2f} MM USD")
    print(f"CRPS h=1-10    [{label}]: {crps_h[:10].mean():.2f} MM USD  (corto plazo)")
    print(f"CRPS h=61-90   [{label}]: {crps_h[60:].mean():.2f} MM USD  (largo plazo)")
    return np.array(hs), crps_h


def winkler_by_h(res_test: pd.DataFrame, banco: str, modelo_id: str,
                 alpha: float = 0.10, guardar: bool = True):
    idx  = res_test["target"].notna()
    df   = res_test[idx].copy()
    hs   = sorted(df["h"].unique())
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

    wink  = np.array(wink)
    color = ESTILO[modelo_id]["color"]
    label = ESTILO[modelo_id]["label"]

    if guardar:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(hs, wink, color=color, alpha=0.5, width=0.8)
        ax.plot(hs, pd.Series(wink).rolling(5, center=True, min_periods=1).mean(),
                color="darkred", lw=2, label="Suavizado 5 días")
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
        ax.set_ylabel("Winkler Score (MM USD)", fontsize=11)
        ax.set_title(f"Winkler Score por h — {banco} [{label}] (TEST)\n"
                     "Intervalo Q05-Q95 (90%)  |  Menor = mejor",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"winkler_by_h_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print(f"\nWinkler medio  [{label}]: {wink.mean():.2f} MM USD")
    return np.array(hs), wink


def cqr_ajuste(res_val: pd.DataFrame, res_test: pd.DataFrame, banco: str,
               modelo_id: str, alpha: float = ALPHA_CQR, guardar: bool = True):
    val_ok = res_val[res_val["target"].notna()].copy()
    if len(val_ok) == 0:
        print("  No hay realizados en VAL — CQR omitido.")
        return None

    s     = np.maximum(val_ok["q05"].values - val_ok["target"].values,
                       val_ok["target"].values - val_ok["q95"].values)
    n_cal = len(s)
    nivel = np.ceil((1 - alpha) * (1 + 1 / n_cal)) / (1 + 1 / n_cal)
    q_hat = np.quantile(s, min(nivel, 1.0))

    label = ESTILO[modelo_id]["label"]
    color = ESTILO[modelo_id]["color"]

    test_ok  = res_test[res_test["target"].notna()].copy()
    y        = test_ok["target"].values
    cov_raw  = ((y >= test_ok["q05"].values) & (y <= test_ok["q95"].values)).mean()
    cov_cqr  = ((y >= test_ok["q05"].values - q_hat) &
                (y <= test_ok["q95"].values + q_hat)).mean()
    ancho_raw = (test_ok["q95"] - test_ok["q05"]).mean() / 1e6
    ancho_cqr = ancho_raw + 2 * q_hat / 1e6

    print(f"\n[{label}] CQR:  n_cal={n_cal}  |  q̂={q_hat/1e6:.1f} MM USD por lado")
    print(f"  Cobertura sin CQR : {cov_raw:.1%}  →  con CQR : {cov_cqr:.1%}  (obj 90%)")
    print(f"  Ancho sin CQR     : {ancho_raw:.1f} MM USD  →  con CQR : {ancho_cqr:.1f} MM USD")

    if guardar:
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
                color=color, lw=2, label=f"Sin CQR ({label})")
        ax.plot(hs, pd.Series(cov_cqr_h).rolling(5, center=True, min_periods=1).mean(),
                color="crimson", lw=2, label=f"Con CQR ({label})")
        ax.scatter(hs, cov_raw_h, color=color,    s=12, alpha=0.35)
        ax.scatter(hs, cov_cqr_h, color="crimson", s=12, alpha=0.35)
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
        ax.set_ylabel("Cobertura empírica", fontsize=11)
        ax.set_title(f"CQR vs Raw — {banco} [{label}]\n"
                     f"q̂ = {q_hat/1e6:.1f} MM USD por lado",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"cqr_coverage_by_h_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    return q_hat, cov_raw, cov_cqr, ancho_raw, ancho_cqr


###############################################################################
# PARTE 3 — COMPARACIÓN LightGBM vs XGBoost
###############################################################################

def comparar_modelos(banco: str, resultados: dict):
    """
    resultados = {
      "lgbm": {"quantiles", "empiricos", "hs", "cov_90", "cov_98",
               "crps_h", "wink_h", "crps_mean", "wink_mean",
               "cov_90_mean", "q_hat", "cov_raw", "cov_cqr"},
      "xgb" : { ... mismas keys ... }
    }
    """
    lgb_r = resultados["lgbm"]
    xgb_r = resultados["xgb"]
    hs    = lgb_r["hs"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    periodo = (f"{CORTE_TEST.strftime('%b %Y')} – {CORTE_TEST_FIN.strftime('%b %Y')}"
               if CORTE_TEST_FIN else f"{CORTE_TEST.strftime('%b %Y')} – hoy")
    fig.suptitle(f"LightGBM vs XGBoost — {banco}  |  TEST: {periodo}\nComparación de calibración",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── 1. Calibration Plot ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Ideal", zorder=0)
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                    alpha=0.08, color="green", label="Tolerancia ±5%")
    for mid in ("lgbm", "xgb"):
        r = resultados[mid]
        ax.plot(r["quantiles"], r["empiricos"], "o-", lw=2, ms=5,
                color=ESTILO[mid]["color"], label=ESTILO[mid]["label"])
    ax.set_xlabel("Cuantil nominal (τ)", fontsize=11)
    ax.set_ylabel("Fracción empírica ≤ Q_τ", fontsize=11)
    ax.set_title("Calibration Plot", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    # ── 2. Coverage Q05-Q95 by h ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.axhline(0.90, color="black", lw=1.2, ls="--", alpha=0.5, label="Objetivo 90%")
    for mid in ("lgbm", "xgb"):
        r     = resultados[mid]
        suav  = pd.Series(r["cov_90"]).rolling(5, center=True, min_periods=1).mean()
        ax.plot(hs, suav, lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} ({r['cov_90_mean']:.1%})")
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("Cobertura Q05-Q95", fontsize=11)
    ax.set_title("Coverage Q05-Q95 por h", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.3)

    # ── 3. CRPS by h ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    for mid in ("lgbm", "xgb"):
        r    = resultados[mid]
        suav = pd.Series(r["crps_h"]).rolling(5, center=True, min_periods=1).mean()
        ax.plot(hs, suav, lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} (media {r['crps_mean']:.1f} MM USD)")
        ax.fill_between(hs, 0, r["crps_h"], alpha=0.12, color=ESTILO[mid]["color"])
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("CRPS (MM USD)", fontsize=11)
    ax.set_title("CRPS por h  (menor = mejor)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.25, axis="y")

    # ── 4. Winkler Score by h ─────────────────────────────────────────────────
    ax = axes[1, 1]
    for mid in ("lgbm", "xgb"):
        r    = resultados[mid]
        suav = pd.Series(r["wink_h"]).rolling(5, center=True, min_periods=1).mean()
        ax.plot(hs, suav, lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} (media {r['wink_mean']:.1f} MM USD)")
        ax.fill_between(hs, 0, r["wink_h"], alpha=0.12, color=ESTILO[mid]["color"])
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("Winkler Score (MM USD)", fontsize=11)
    ax.set_title("Winkler Score por h  (menor = mejor)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / f"comparacion_{banco}_lgbm_vs_xgb.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGuardado: comparacion_{banco}_lgbm_vs_xgb.png")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print(f"  RESUMEN COMPARATIVO — {banco} (TEST)")
    print("="*68)
    print(f"  {'Métrica':<35}  {'LightGBM':>12}  {'XGBoost':>12}  {'Ganador'}")
    print("-"*68)

    def fila(nombre, val_lgb, val_xgb, fmt="{:.2f}", menor_es_mejor=True):
        s_lgb = fmt.format(val_lgb)
        s_xgb = fmt.format(val_xgb)
        if menor_es_mejor:
            gan = "LightGBM ✓" if val_lgb < val_xgb else "XGBoost ✓" if val_xgb < val_lgb else "Empate"
        else:
            gan = "LightGBM ✓" if val_lgb > val_xgb else "XGBoost ✓" if val_xgb > val_lgb else "Empate"
        print(f"  {nombre:<35}  {s_lgb:>12}  {s_xgb:>12}  {gan}")

    fila("CRPS medio (MM USD)",
         lgb_r["crps_mean"], xgb_r["crps_mean"], menor_es_mejor=True)
    fila("CRPS h=1-10 (MM USD)",
         lgb_r["crps_h"][:10].mean(), xgb_r["crps_h"][:10].mean(), menor_es_mejor=True)
    fila("CRPS h=61-90 (MM USD)",
         lgb_r["crps_h"][60:].mean(), xgb_r["crps_h"][60:].mean(), menor_es_mejor=True)
    fila("Winkler medio (MM USD)",
         lgb_r["wink_mean"], xgb_r["wink_mean"], menor_es_mejor=True)
    fila("Cobertura Q05-Q95 (%)",
         lgb_r["cov_90_mean"]*100, xgb_r["cov_90_mean"]*100,
         fmt="{:.1f}", menor_es_mejor=False)
    fila("Cobertura Q01-Q99 (%)",
         lgb_r["cov_98_mean"]*100, xgb_r["cov_98_mean"]*100,
         fmt="{:.1f}", menor_es_mejor=False)
    fila("CQR q̂ (MM USD por lado)",
         lgb_r["q_hat"]/1e6, xgb_r["q_hat"]/1e6, menor_es_mejor=True)
    fila("Cobertura CQR (%)",
         lgb_r["cov_cqr"]*100, xgb_r["cov_cqr"]*100,
         fmt="{:.1f}", menor_es_mejor=False)

    print("="*68)

    # ── Veredicto ─────────────────────────────────────────────────────────────
    puntos = {"lgbm": 0, "xgb": 0}
    metricas_comp = [
        (lgb_r["crps_mean"],          xgb_r["crps_mean"],          True),
        (lgb_r["wink_mean"],          xgb_r["wink_mean"],          True),
        (lgb_r["cov_90_mean"],        xgb_r["cov_90_mean"],        False),
        (lgb_r["q_hat"],              xgb_r["q_hat"],              True),
    ]
    for v_lgb, v_xgb, menor in metricas_comp:
        mejor = "lgbm" if (v_lgb < v_xgb) == menor else "xgb"
        puntos[mejor] += 1

    ganador = "LightGBM" if puntos["lgbm"] > puntos["xgb"] else \
              "XGBoost"  if puntos["xgb"] > puntos["lgbm"] else "Empate"
    print(f"\n  Puntos: LightGBM={puntos['lgbm']}  XGBoost={puntos['xgb']}")
    print(f"  → Mejor performance general: {ganador}")
    print("="*68)


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    resultados = {}

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print("PARTE 1 — LightGBM")
    print("="*68)
    modelos_lgb, cols_lgb, quantiles_lgb = cargar_modelos_lgbm(BANCO, DIR_MOD_LGB)
    df_val_lgb, df_test_lgb = cargar_splits(BANCO, cols_lgb)

    print("\nPrediciendo con LightGBM...")
    res_val_lgb  = predecir_df_lgbm(modelos_lgb, df_val_lgb,  cols_lgb)
    res_test_lgb = predecir_df_lgbm(modelos_lgb, df_test_lgb, cols_lgb)

    emp_lgb             = calibration_plot(res_test_lgb, quantiles_lgb, BANCO, "lgbm")
    hs, cov90_lgb, cov98_lgb = coverage_by_h(res_test_lgb, BANCO, "lgbm")
    _, crps_lgb         = crps_by_h(res_test_lgb, quantiles_lgb, BANCO, "lgbm")
    _, wink_lgb         = winkler_by_h(res_test_lgb, BANCO, "lgbm")
    q_hat_lgb, cov_raw_lgb, cov_cqr_lgb, ancho_raw_lgb, _ = cqr_ajuste(
        res_val_lgb, res_test_lgb, BANCO, "lgbm")

    resultados["lgbm"] = {
        "quantiles" : quantiles_lgb,
        "empiricos" : emp_lgb,
        "hs"        : hs,
        "cov_90"    : cov90_lgb,
        "cov_98"    : cov98_lgb,
        "cov_90_mean": np.nanmean(cov90_lgb),
        "cov_98_mean": np.nanmean(cov98_lgb),
        "crps_h"    : crps_lgb,
        "crps_mean" : crps_lgb.mean(),
        "wink_h"    : wink_lgb,
        "wink_mean" : wink_lgb.mean(),
        "q_hat"     : q_hat_lgb,
        "cov_raw"   : cov_raw_lgb,
        "cov_cqr"   : cov_cqr_lgb,
    }

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print("PARTE 2 — XGBoost")
    print("="*68)
    modelos_xgb, cols_xgb, quantiles_xgb = cargar_modelos_xgb(BANCO, DIR_MOD_XGB)
    df_val_xgb, df_test_xgb = cargar_splits(BANCO, cols_xgb)

    print("\nPrediciendo con XGBoost...")
    res_val_xgb  = predecir_df_xgb(modelos_xgb, df_val_xgb,  cols_xgb)
    res_test_xgb = predecir_df_xgb(modelos_xgb, df_test_xgb, cols_xgb)

    emp_xgb                  = calibration_plot(res_test_xgb, quantiles_xgb, BANCO, "xgb")
    _, cov90_xgb, cov98_xgb  = coverage_by_h(res_test_xgb, BANCO, "xgb")
    _, crps_xgb              = crps_by_h(res_test_xgb, quantiles_xgb, BANCO, "xgb")
    _, wink_xgb              = winkler_by_h(res_test_xgb, BANCO, "xgb")
    q_hat_xgb, cov_raw_xgb, cov_cqr_xgb, ancho_raw_xgb, _ = cqr_ajuste(
        res_val_xgb, res_test_xgb, BANCO, "xgb")

    resultados["xgb"] = {
        "quantiles" : quantiles_xgb,
        "empiricos" : emp_xgb,
        "hs"        : hs,
        "cov_90"    : cov90_xgb,
        "cov_98"    : cov98_xgb,
        "cov_90_mean": np.nanmean(cov90_xgb),
        "cov_98_mean": np.nanmean(cov98_xgb),
        "crps_h"    : crps_xgb,
        "crps_mean" : crps_xgb.mean(),
        "wink_h"    : wink_xgb,
        "wink_mean" : wink_xgb.mean(),
        "q_hat"     : q_hat_xgb,
        "cov_raw"   : cov_raw_xgb,
        "cov_cqr"   : cov_cqr_xgb,
    }

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print("PARTE 3 — COMPARACIÓN LightGBM vs XGBoost")
    print("="*68)
    comparar_modelos(BANCO, resultados)

    print(f"\n✓ Todos los outputs en: {DIR_OUTPUT}")
