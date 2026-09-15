# -*- coding: utf-8 -*-
"""
calibracion_v1.py
Diagnóstico de calibración comparativo: XGBoost (step004) vs XGBoost QT (step004_qt).

Estructura:
  PARTE 1 — XGBoost    : modelos_xgb/eval/        | darkorange
  PARTE 2 — XGBoost QT : modelos_xgb_qt/eval/     | tomato
  PARTE 3 — Comparación: plots superpuestos + tabla resumen + veredicto

Métricas:
  1. Calibration Plot   : cobertura empírica vs nominal Q01…Q99
  2. Coverage by h      : % dentro de Q05-Q95 y Q01-Q99 por horizonte
  3. CRPS by h          : Continuous Ranked Probability Score por horizonte
  4. Winkler Score by h : sharpness + penalización por salidas del intervalo
  5. CQR                : ajuste de garantía finita usando VAL como calibración
  6. Sharpness by h     : ancho medio del intervalo Q05-Q95 (menor = más preciso)

XGBoost estándar  → step004_train_xgboost.py   → modelos_xgb/eval/
XGBoost QT        → step004_train_xgboost_qt.py → modelos_xgb_qt/eval/
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_SISTEMA    = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ     = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MOD_XGB     = BASE_SISTEMA / "2. Output" / "modelos_xgb"    / "eval"
DIR_MOD_XGB_QT  = BASE_SISTEMA / "2. Output" / "modelos_xgb_qt" / "eval"
DIR_OUTPUT      = BASE_SISTEMA / "2. Output" / "calibracion_v1"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO          = "SISTEMA"
CORTE_VAL      = pd.Timestamp("2022-07-01")
CORTE_TEST     = pd.Timestamp("2023-01-03")
CORTE_TEST_FIN = None   # None = toda la muestra TEST | ej: pd.Timestamp("2023-12-31")
ALPHA_CQR      = 0.10   # cobertura objetivo = 1 - ALPHA_CQR = 90 %

# Paleta consistente con los fancharts
ESTILO = {
    "xgb"    : {"color": "darkorange", "label": "XGBoost (step004)"},
    "xgb_qt" : {"color": "tomato",     "label": "XGBoost QT (step004_qt)"},
}


###############################################################################
# CARGA DE MODELOS
###############################################################################

def _cargar_xgb_generico(banco: str, dir_modelos: Path,
                          glob_meta: str, glob_modelo: str,
                          modelo_id: str) -> tuple[dict, list, list]:
    metas = sorted(dir_modelos.glob(glob_meta), reverse=True)
    if not metas:
        raise FileNotFoundError(
            f"No se encontró metadata {modelo_id} para banco={banco} en {dir_modelos}"
        )
    meta      = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha     = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / glob_modelo.format(banco=banco,
                                                 q=int(tau * 100),
                                                 fecha=fecha)
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        booster = xgb.Booster()
        booster.load_model(str(ruta))
        modelos[tau] = booster

    label = ESTILO[modelo_id]["label"]
    print(f"[{label}]  {metas[0].name}")
    print(f"  Quantiles : {quantiles}  |  Features : {len(cols_feat)}")
    return modelos, cols_feat, quantiles


def cargar_modelos_xgb(banco: str, dir_modelos: Path):
    return _cargar_xgb_generico(
        banco, dir_modelos,
        glob_meta   = f"metadata_xgb_{banco}_*.json",
        glob_modelo = "xgb_{banco}_q{q:02d}_{fecha}.json",
        modelo_id   = "xgb",
    )


def cargar_modelos_xgb_qt(banco: str, dir_modelos: Path):
    return _cargar_xgb_generico(
        banco, dir_modelos,
        glob_meta   = f"metadata_xgb_qt_{banco}_*.json",
        glob_modelo = "xgb_qt_{banco}_q{q:02d}_{fecha}.json",
        modelo_id   = "xgb_qt",
    )


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
    mask    = df["fecha_t"] >= CORTE_TEST
    if CORTE_TEST_FIN is not None:
        mask &= df["fecha_t"] <= CORTE_TEST_FIN
    df_test = imputar(df[mask])

    print(f"  VAL : {CORTE_VAL.date()} → {df_val['fecha_t'].max().date()}"
          f"  ({df_val['fecha_t'].nunique()} fechas)")
    fin_test = CORTE_TEST_FIN.date() if CORTE_TEST_FIN else df_test["fecha_t"].max().date()
    print(f"  TEST: {CORTE_TEST.date()} → {fin_test}"
          f"  ({df_test['fecha_t'].nunique()} fechas)"
          f"{'  [primer año]' if CORTE_TEST_FIN else ''}")
    return df_val, df_test


###############################################################################
# PREDICCIÓN
###############################################################################

def predecir_df(modelos: dict, df: pd.DataFrame, cols_feat: list[str]) -> pd.DataFrame:
    cols_ok = [c for c in cols_feat if c in df.columns]
    dmat    = xgb.DMatrix(df[cols_ok].copy())
    res     = df[["fecha_t", "h", "target"]].copy().reset_index(drop=True)
    for tau, m in modelos.items():
        res[f"q{int(tau*100):02d}"] = m.predict(dmat)
    q_cols = sorted(c for c in res.columns if c.startswith("q"))
    res[q_cols] = np.sort(res[q_cols].values, axis=1)
    return res


###############################################################################
# MÉTRICAS
###############################################################################

def calibration_plot(res_test: pd.DataFrame, quantiles: list,
                     banco: str, modelo_id: str, guardar: bool = True):
    idx       = res_test["target"].notna()
    empiricos = []
    for tau in quantiles:
        col  = f"q{int(tau*100):02d}"
        frac = (res_test.loc[idx, "target"].values <=
                res_test.loc[idx, col].values).mean()
        empiricos.append(frac)

    color = ESTILO[modelo_id]["color"]
    label = ESTILO[modelo_id]["label"]

    if guardar:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(quantiles, empiricos, "o-", lw=2, ms=6, color=color,
                label=f"Empírico ({label})")
        ax.plot([0, 1], [0, 1], "r--", lw=2, label="Calibración perfecta")
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                        alpha=0.10, color="green", label="Tolerancia ±5%")
        ax.set_xlabel("Cuantil nominal (τ)", fontsize=12)
        ax.set_ylabel("Fracción empírica ≤ Q_τ", fontsize=12)
        ax.set_title(f"Calibration Plot — {banco} [{label}] (TEST)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"calibration_plot_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print(f"\n  τ_nominal  τ_empírico   Δ   [{label}]")
    for tau, emp in zip(quantiles, empiricos):
        delta = emp - tau
        flag  = "  ←" if abs(delta) > 0.05 else ""
        print(f"  {tau:.2f}       {emp:.3f}      {delta:+.3f}{flag}")

    return empiricos


def coverage_by_h(res_test: pd.DataFrame, banco: str,
                  modelo_id: str, guardar: bool = True):
    idx = res_test["target"].notna()
    df  = res_test[idx].copy()
    hs  = sorted(df["h"].unique())
    cov_90, cov_98 = [], []

    for h in hs:
        sub = df[df["h"] == h]
        y   = sub["target"].values
        cov_90.append(((y >= sub["q05"].values) & (y <= sub["q95"].values)).mean())
        cov_98.append(((y >= sub["q01"].values) & (y <= sub["q99"].values)).mean())

    hs_arr     = np.array(hs)
    cov_90_arr = np.array(cov_90)
    cov_98_arr = np.array(cov_98)
    color      = ESTILO[modelo_id]["color"]
    label      = ESTILO[modelo_id]["label"]

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
        ax.scatter(hs_arr, cov_98_arr, color="navy",  s=14, alpha=0.4)
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
        lower = sub["q05"].values    / 1e6
        upper = sub["q95"].values    / 1e6
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


def sharpness_by_h(res_test: pd.DataFrame, banco: str,
                   modelo_id: str, guardar: bool = True):
    """Ancho medio del intervalo Q05-Q95 por horizonte (sharpness; menor = más preciso)."""
    df  = res_test.copy()
    hs  = sorted(df["h"].unique())
    sharp = []

    for h in hs:
        sub = df[df["h"] == h]
        ancho = (sub["q95"].values - sub["q05"].values) / 1e6
        sharp.append(ancho.mean())

    sharp = np.array(sharp)
    color = ESTILO[modelo_id]["color"]
    label = ESTILO[modelo_id]["label"]

    if guardar:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(hs, sharp, color=color, alpha=0.5, width=0.8)
        ax.plot(hs, pd.Series(sharp).rolling(5, center=True, min_periods=1).mean(),
                color="black", lw=2, label="Suavizado 5 días")
        ax.set_xlabel("Horizonte h (días hábiles)", fontsize=11)
        ax.set_ylabel("Ancho Q05-Q95 (MM USD)", fontsize=11)
        ax.set_title(f"Sharpness (ancho intervalo) por h — {banco} [{label}] (TEST)\n"
                     "Menor = bandas más estrechas",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(DIR_OUTPUT / f"sharpness_by_h_{banco}_{modelo_id}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    print(f"\nAncho medio Q05-Q95 [{label}]: {sharp.mean():.2f} MM USD")
    print(f"  h=1-10  : {sharp[:10].mean():.2f} MM USD")
    print(f"  h=61-90 : {sharp[60:].mean():.2f} MM USD")
    return np.array(hs), sharp


def cqr_ajuste(res_val: pd.DataFrame, res_test: pd.DataFrame,
               banco: str, modelo_id: str, alpha: float = ALPHA_CQR,
               guardar: bool = True):
    val_ok = res_val[res_val["target"].notna()].copy()
    if len(val_ok) == 0:
        print("  No hay realizados en VAL — CQR omitido.")
        return None, None, None, None, None

    s     = np.maximum(val_ok["q05"].values - val_ok["target"].values,
                       val_ok["target"].values - val_ok["q95"].values)
    n_cal = len(s)
    nivel = np.ceil((1 - alpha) * (1 + 1 / n_cal)) / (1 + 1 / n_cal)
    q_hat = np.quantile(s, min(nivel, 1.0))

    label = ESTILO[modelo_id]["label"]
    color = ESTILO[modelo_id]["color"]

    test_ok   = res_test[res_test["target"].notna()].copy()
    y         = test_ok["target"].values
    cov_raw   = ((y >= test_ok["q05"].values) & (y <= test_ok["q95"].values)).mean()
    cov_cqr   = ((y >= test_ok["q05"].values - q_hat) &
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
            cov_raw_h.append(((yh >= sub["q05"].values) &
                               (yh <= sub["q95"].values)).mean())
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
# PARTE 3 — COMPARACIÓN XGBoost vs XGBoost QT
###############################################################################

def comparar_modelos(banco: str, resultados: dict):
    """
    resultados = {
      "xgb"    : {quantiles, empiricos, hs, cov_90, cov_98, crps_h, wink_h,
                  sharp_h, crps_mean, wink_mean, sharp_mean,
                  cov_90_mean, cov_98_mean, q_hat, cov_raw, cov_cqr},
      "xgb_qt" : { ... mismas keys ... }
    }
    """
    r_xgb    = resultados["xgb"]
    r_xgb_qt = resultados["xgb_qt"]
    hs       = r_xgb["hs"]

    def smooth(x, w=5):
        return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values

    periodo = (f"{CORTE_TEST.strftime('%b %Y')} – {CORTE_TEST_FIN.strftime('%b %Y')}"
               if CORTE_TEST_FIN else f"{CORTE_TEST.strftime('%b %Y')} – hoy")

    # ── Panel 1: 2×3 — Calibration + Coverage + CRPS + Winkler + Sharpness + CQR ──
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"XGBoost (step004) vs XGBoost QT (step004_qt) — {banco}  |  TEST: {periodo}\n"
        "Diagnóstico comparativo de calibración",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # 1. Calibration Plot
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Ideal", zorder=0)
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                    alpha=0.08, color="green", label="Tolerancia ±5%")
    for mid in ("xgb", "xgb_qt"):
        r = resultados[mid]
        ax.plot(r["quantiles"], r["empiricos"], "o-", lw=2, ms=5,
                color=ESTILO[mid]["color"], label=ESTILO[mid]["label"])
    ax.set_xlabel("Cuantil nominal (τ)", fontsize=11)
    ax.set_ylabel("Fracción empírica ≤ Q_τ", fontsize=11)
    ax.set_title("Calibration Plot", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    # 2. Coverage Q05-Q95 by h
    ax = axes[0, 1]
    ax.axhline(0.90, color="black", lw=1.2, ls="--", alpha=0.5, label="Objetivo 90%")
    for mid in ("xgb", "xgb_qt"):
        r = resultados[mid]
        ax.plot(hs, smooth(r["cov_90"]), lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} ({r['cov_90_mean']:.1%})")
        ax.fill_between(hs, r["cov_90"], 0.90,
                        where=np.array(r["cov_90"]) < 0.90,
                        alpha=0.08, color=ESTILO[mid]["color"])
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("Cobertura Q05-Q95", fontsize=11)
    ax.set_title("Coverage Q05-Q95 por h", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.3)

    # 3. CRPS by h
    ax = axes[0, 2]
    for mid in ("xgb", "xgb_qt"):
        r = resultados[mid]
        ax.plot(hs, smooth(r["crps_h"]), lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} ({r['crps_mean']:.1f} MM USD)")
        ax.fill_between(hs, 0, r["crps_h"], alpha=0.12, color=ESTILO[mid]["color"])
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("CRPS (MM USD)", fontsize=11)
    ax.set_title("CRPS por h  (menor = mejor)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.25, axis="y")

    # 4. Winkler Score by h
    ax = axes[1, 0]
    for mid in ("xgb", "xgb_qt"):
        r = resultados[mid]
        ax.plot(hs, smooth(r["wink_h"]), lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} ({r['wink_mean']:.1f} MM USD)")
        ax.fill_between(hs, 0, r["wink_h"], alpha=0.12, color=ESTILO[mid]["color"])
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("Winkler Score (MM USD)", fontsize=11)
    ax.set_title("Winkler Score por h  (menor = mejor)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.25, axis="y")

    # 5. Sharpness by h
    ax = axes[1, 1]
    for mid in ("xgb", "xgb_qt"):
        r = resultados[mid]
        ax.plot(hs, smooth(r["sharp_h"]), lw=2, color=ESTILO[mid]["color"],
                label=f"{ESTILO[mid]['label']} ({r['sharp_mean']:.1f} MM USD)")
    ax.set_xlabel("Horizonte h", fontsize=11)
    ax.set_ylabel("Ancho Q05-Q95 (MM USD)", fontsize=11)
    ax.set_title("Sharpness (ancho intervalo) por h\nMenor = bandas más estrechas",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10)); ax.grid(alpha=0.25, axis="y")

    # 6. CQR coverage by h (raw vs CQR para ambos modelos)
    ax = axes[1, 2]
    ax.axhline(1 - ALPHA_CQR, color="green", lw=1.5, ls="--", alpha=0.8,
               label=f"Objetivo {1-ALPHA_CQR:.0%}")
    for mid in ("xgb", "xgb_qt"):
        r = resultados[mid]
        ax.axhline(r["cov_cqr"], color=ESTILO[mid]["color"], lw=1.5, ls=":",
                   alpha=0.7,
                   label=f"CQR {ESTILO[mid]['label']} ({r['cov_cqr']:.1%})")
        ax.axhline(r["cov_raw"], color=ESTILO[mid]["color"], lw=1.5, ls="-",
                   alpha=0.7,
                   label=f"Raw {ESTILO[mid]['label']} ({r['cov_raw']:.1%})")
    ax.set_xlabel("—", fontsize=10)
    ax.set_ylabel("Cobertura global", fontsize=11)
    ax.set_title("Cobertura global: Raw vs CQR\n(línea sólida = raw, punteada = CQR)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0)); ax.grid(alpha=0.3)

    plt.tight_layout()
    nombre_comp = f"comparacion_{banco}_xgb_vs_xgb_qt.png"
    plt.savefig(DIR_OUTPUT / nombre_comp, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGuardado: {nombre_comp}")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  RESUMEN COMPARATIVO — {banco}  |  TEST: {periodo}")
    print("=" * 72)
    print(f"  {'Métrica':<38}  {'XGBoost':>12}  {'XGBoost QT':>12}  {'Ganador'}")
    print("-" * 72)

    def fila(nombre, val_xgb, val_qt, fmt="{:.2f}", menor_es_mejor=True):
        s_xgb = fmt.format(val_xgb)
        s_qt  = fmt.format(val_qt)
        if menor_es_mejor:
            gan = "XGBoost ✓" if val_xgb < val_qt else \
                  "XGBoost QT ✓" if val_qt < val_xgb else "Empate"
        else:
            gan = "XGBoost ✓" if val_xgb > val_qt else \
                  "XGBoost QT ✓" if val_qt > val_xgb else "Empate"
        print(f"  {nombre:<38}  {s_xgb:>12}  {s_qt:>12}  {gan}")

    r_x = resultados["xgb"]
    r_q = resultados["xgb_qt"]

    fila("CRPS medio (MM USD)",           r_x["crps_mean"],        r_q["crps_mean"])
    fila("CRPS h=1-10 (MM USD)",          r_x["crps_h"][:10].mean(), r_q["crps_h"][:10].mean())
    fila("CRPS h=61-90 (MM USD)",         r_x["crps_h"][60:].mean(), r_q["crps_h"][60:].mean())
    fila("Winkler medio (MM USD)",        r_x["wink_mean"],        r_q["wink_mean"])
    fila("Sharpness media (MM USD)",      r_x["sharp_mean"],       r_q["sharp_mean"])
    fila("Cobertura Q05-Q95 (%)",
         r_x["cov_90_mean"] * 100, r_q["cov_90_mean"] * 100,
         fmt="{:.1f}", menor_es_mejor=False)
    fila("Cobertura Q01-Q99 (%)",
         r_x["cov_98_mean"] * 100, r_q["cov_98_mean"] * 100,
         fmt="{:.1f}", menor_es_mejor=False)
    fila("CQR q̂ (MM USD por lado)",
         r_x["q_hat"] / 1e6,  r_q["q_hat"] / 1e6)
    fila("Cobertura sin CQR (%)",
         r_x["cov_raw"] * 100, r_q["cov_raw"] * 100,
         fmt="{:.1f}", menor_es_mejor=False)
    fila("Cobertura con CQR (%)",
         r_x["cov_cqr"] * 100, r_q["cov_cqr"] * 100,
         fmt="{:.1f}", menor_es_mejor=False)
    print("=" * 72)

    # ── Veredicto ─────────────────────────────────────────────────────────────
    puntos = {"xgb": 0, "xgb_qt": 0}
    criterios = [
        (r_x["crps_mean"],          r_q["crps_mean"],          True),
        (r_x["wink_mean"],          r_q["wink_mean"],          True),
        (r_x["sharp_mean"],         r_q["sharp_mean"],         True),
        (r_x["cov_90_mean"],        r_q["cov_90_mean"],        False),
        (r_x["q_hat"],              r_q["q_hat"],              True),
    ]
    for v_x, v_q, menor in criterios:
        if menor:
            gan = "xgb" if v_x < v_q else "xgb_qt" if v_q < v_x else None
        else:
            gan = "xgb" if v_x > v_q else "xgb_qt" if v_q > v_x else None
        if gan:
            puntos[gan] += 1

    ganador = ("XGBoost (step004)"    if puntos["xgb"] > puntos["xgb_qt"] else
               "XGBoost QT (step004_qt)" if puntos["xgb_qt"] > puntos["xgb"] else
               "Empate")
    print(f"\n  Puntos: XGBoost={puntos['xgb']}  XGBoost QT={puntos['xgb_qt']}  "
          f"(criterios: CRPS, Winkler, Sharpness, Cobertura Q05-Q95, CQR q̂)")
    print(f"  → Mejor performance general: {ganador}")
    print("=" * 72)


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    resultados = {}

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("PARTE 1 — XGBoost (step004_train_xgboost.py)")
    print("=" * 72)
    modelos_xgb, cols_xgb, quantiles_xgb = cargar_modelos_xgb(BANCO, DIR_MOD_XGB)
    df_val_xgb, df_test_xgb = cargar_splits(BANCO, cols_xgb)

    print("\nPrediciendo con XGBoost...")
    res_val_xgb  = predecir_df(modelos_xgb, df_val_xgb,  cols_xgb)
    res_test_xgb = predecir_df(modelos_xgb, df_test_xgb, cols_xgb)

    emp_xgb                      = calibration_plot(res_test_xgb, quantiles_xgb, BANCO, "xgb")
    hs, cov90_xgb, cov98_xgb     = coverage_by_h(res_test_xgb, BANCO, "xgb")
    _, crps_xgb                  = crps_by_h(res_test_xgb, quantiles_xgb, BANCO, "xgb")
    _, wink_xgb                  = winkler_by_h(res_test_xgb, BANCO, "xgb")
    _, sharp_xgb                 = sharpness_by_h(res_test_xgb, BANCO, "xgb")
    q_hat_xgb, cov_raw_xgb, cov_cqr_xgb, *_ = cqr_ajuste(
        res_val_xgb, res_test_xgb, BANCO, "xgb")

    resultados["xgb"] = {
        "quantiles"   : quantiles_xgb,
        "empiricos"   : emp_xgb,
        "hs"          : hs,
        "cov_90"      : cov90_xgb,
        "cov_98"      : cov98_xgb,
        "cov_90_mean" : np.nanmean(cov90_xgb),
        "cov_98_mean" : np.nanmean(cov98_xgb),
        "crps_h"      : crps_xgb,
        "crps_mean"   : crps_xgb.mean(),
        "wink_h"      : wink_xgb,
        "wink_mean"   : wink_xgb.mean(),
        "sharp_h"     : sharp_xgb,
        "sharp_mean"  : sharp_xgb.mean(),
        "q_hat"       : q_hat_xgb,
        "cov_raw"     : cov_raw_xgb,
        "cov_cqr"     : cov_cqr_xgb,
    }

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("PARTE 2 — XGBoost QT (step004_train_xgboost_qt.py)")
    print("=" * 72)
    modelos_qt, cols_qt, quantiles_qt = cargar_modelos_xgb_qt(BANCO, DIR_MOD_XGB_QT)
    df_val_qt, df_test_qt = cargar_splits(BANCO, cols_qt)

    print("\nPrediciendo con XGBoost QT...")
    res_val_qt  = predecir_df(modelos_qt, df_val_qt,  cols_qt)
    res_test_qt = predecir_df(modelos_qt, df_test_qt, cols_qt)

    emp_qt                      = calibration_plot(res_test_qt, quantiles_qt, BANCO, "xgb_qt")
    _, cov90_qt, cov98_qt       = coverage_by_h(res_test_qt, BANCO, "xgb_qt")
    _, crps_qt                  = crps_by_h(res_test_qt, quantiles_qt, BANCO, "xgb_qt")
    _, wink_qt                  = winkler_by_h(res_test_qt, BANCO, "xgb_qt")
    _, sharp_qt                 = sharpness_by_h(res_test_qt, BANCO, "xgb_qt")
    q_hat_qt, cov_raw_qt, cov_cqr_qt, *_ = cqr_ajuste(
        res_val_qt, res_test_qt, BANCO, "xgb_qt")

    resultados["xgb_qt"] = {
        "quantiles"   : quantiles_qt,
        "empiricos"   : emp_qt,
        "hs"          : hs,
        "cov_90"      : cov90_qt,
        "cov_98"      : cov98_qt,
        "cov_90_mean" : np.nanmean(cov90_qt),
        "cov_98_mean" : np.nanmean(cov98_qt),
        "crps_h"      : crps_qt,
        "crps_mean"   : crps_qt.mean(),
        "wink_h"      : wink_qt,
        "wink_mean"   : wink_qt.mean(),
        "sharp_h"     : sharp_qt,
        "sharp_mean"  : sharp_qt.mean(),
        "q_hat"       : q_hat_qt,
        "cov_raw"     : cov_raw_qt,
        "cov_cqr"     : cov_cqr_qt,
    }

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("PARTE 3 — COMPARACIÓN XGBoost vs XGBoost QT")
    print("=" * 72)
    comparar_modelos(BANCO, resultados)

    print(f"\n✓ Todos los outputs en: {DIR_OUTPUT}")
