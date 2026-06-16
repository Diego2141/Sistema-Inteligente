# -*- coding: utf-8 -*-
"""
aux_comparar_prophet.py
=======================
Script auxiliar que ajusta Prophet banco por banco y fold por fold,
y grafica sus predicciones contra la realidad.

Permite diagnosticar visualmente qué captura Prophet
(tendencia + estacionalidad) antes de combinarlo con XGB.

Salidas por fold:
  - Gráfico: yhat ± intervalo vs flujo real (TEST y VAL)
  - Gráfico: componentes de Prophet (tendencia, semana, año, trimestre)
  - CSV resumen: pinball_q50, cobertura 80% fold × banco

Uso:
    python aux_comparar_prophet.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False
    raise ImportError("Instala prophet: pip install prophet")

# ── Configuración ──────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_prophet_comparar"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

# Bancos a analizar  (None = todos los disponibles)
BANCOS       = ["SISTEMA"]

# Folds walk-forward (mismas fechas que step005_walk_forward_cv_5)
H_REF        = 2    # h para extraer flujo neto diario

FOLDS = [
    {"fold": 1, "train_start": "2010-01-04", "train_end": "2015-12-31",
                "val_start":   "2016-01-04", "val_end":   "2016-12-30",
                "test_start":  "2017-01-02", "test_end":  "2017-12-29"},
    {"fold": 2, "train_start": "2010-01-04", "train_end": "2016-12-30",
                "val_start":   "2017-01-02", "val_end":   "2017-12-29",
                "test_start":  "2018-01-02", "test_end":  "2018-12-31"},
    {"fold": 3, "train_start": "2010-01-04", "train_end": "2017-12-29",
                "val_start":   "2018-01-02", "val_end":   "2018-12-31",
                "test_start":  "2019-01-02", "test_end":  "2019-12-31"},
    {"fold": 4, "train_start": "2010-01-04", "train_end": "2018-12-31",
                "val_start":   "2019-01-02", "val_end":   "2019-12-31",
                "test_start":  "2020-01-02", "test_end":  "2020-12-31"},
    {"fold": 5, "train_start": "2010-01-04", "train_end": "2019-12-31",
                "val_start":   "2020-01-02", "val_end":   "2020-12-31",
                "test_start":  "2021-01-04", "test_end":  "2021-12-31"},
    {"fold": 6, "train_start": "2010-01-04", "train_end": "2020-12-31",
                "val_start":   "2021-01-04", "val_end":   "2021-12-31",
                "test_start":  "2022-01-03", "test_end":  "2022-12-30"},
    {"fold": 7, "train_start": "2010-01-04", "train_end": "2021-12-31",
                "val_start":   "2022-01-03", "val_end":   "2022-12-30",
                "test_start":  "2023-01-02", "test_end":  "2023-12-29"},
    {"fold": 8, "train_start": "2010-01-04", "train_end": "2022-12-30",
                "val_start":   "2023-01-02", "val_end":   "2023-12-29",
                "test_start":  "2024-01-02", "test_end":  "2024-12-31"},
]

PROPHET_CHANGEPOINTS      = ['2016-07-01']
PROPHET_CHANGEPOINT_PRIOR = 0.05

# ── Carga de datos ─────────────────────────────────────────────────────────────

def cargar(banco: str, h_ref: int = H_REF) -> pd.DataFrame:
    df = pd.read_parquet(
        RUTA_MATRIZ,
        filters=[("banco", "==", banco), ("h", "==", h_ref)]
    )
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    return df.sort_values("fecha_t").reset_index(drop=True)


# ── Prophet helpers ────────────────────────────────────────────────────────────

def ajustar_prophet(fechas_train, y_train_vals):
    import logging as _lg
    _lg.getLogger("cmdstanpy").setLevel(_lg.WARNING)
    _lg.getLogger("prophet").setLevel(_lg.WARNING)

    df_p = pd.DataFrame({"ds": pd.to_datetime(fechas_train),
                          "y":  y_train_vals}).dropna()
    cp_valid = [c for c in PROPHET_CHANGEPOINTS
                if df_p["ds"].min() < pd.Timestamp(c) < df_p["ds"].max()]
    m = Prophet(
        changepoints=cp_valid if cp_valid else None,
        changepoint_prior_scale=PROPHET_CHANGEPOINT_PRIOR,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.80,
    )
    m.add_seasonality(name="monthly",   period=21, fourier_order=5)
    m.add_seasonality(name="quarterly", period=63, fourier_order=5)
    m.fit(df_p)
    return m


def predecir(model, fechas):
    future = pd.DataFrame({"ds": pd.to_datetime(fechas)})
    return model.predict(future)


# ── Métricas ───────────────────────────────────────────────────────────────────

def pinball(y_true, y_pred, tau):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def cobertura(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# ── Gráfico fold ───────────────────────────────────────────────────────────────

def graficar_fold(fold_num, banco, df_train, df_test, df_val, fc_test, fc_val,
                  model):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax = axes[0]

    # Últimos 2 años de train + val + test
    train_tail = df_train[df_train["fecha_t"] >= df_train["fecha_t"].max()
                          - pd.DateOffset(years=2)]

    # Realidad
    ax.plot(train_tail["fecha_t"], train_tail["target"] / 1e9,
            color="silver", lw=0.7, label="Train (últimos 2a)")
    ax.plot(df_val["fecha_t"],  df_val["target"]  / 1e9,
            color="steelblue", lw=1.2, label="Real VAL")
    ax.plot(df_test["fecha_t"], df_test["target"] / 1e9,
            color="navy",      lw=1.4, label="Real TEST")

    # Predicciones Prophet
    ax.plot(fc_val["ds"],  fc_val["yhat"]  / 1e9, "--", color="darkorange",
            lw=1.2, label="Prophet VAL")
    ax.fill_between(fc_val["ds"], fc_val["yhat_lower"] / 1e9,
                    fc_val["yhat_upper"] / 1e9, alpha=0.15, color="darkorange")
    ax.plot(fc_test["ds"],  fc_test["yhat"] / 1e9, "--", color="red",
            lw=1.4, label="Prophet TEST")
    ax.fill_between(fc_test["ds"], fc_test["yhat_lower"] / 1e9,
                    fc_test["yhat_upper"] / 1e9, alpha=0.15, color="red")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Flujo neto (miles de millones S/.)")
    ax.set_title(f"Prophet puro vs realidad — {banco}  |  Fold {fold_num}  "
                 f"|  TRAIN hasta {df_train['fecha_t'].max().date()}", fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Residuo TEST
    ax2 = axes[1]
    resid_test = (df_test.set_index("fecha_t")["target"]
                  .reindex(pd.DatetimeIndex(fc_test["ds"]))
                  - fc_test.set_index("ds")["yhat"]) / 1e9
    ax2.bar(resid_test.index, resid_test.values,
            color=np.where(resid_test.values >= 0, "steelblue", "salmon"),
            width=1.0, alpha=0.7)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("Residuo TEST\n(miles de mill.)", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ruta = DIR_OUTPUT / f"prophet_{banco}_fold{fold_num:02d}.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")


def graficar_componentes(fold_num, banco, model, df_train):
    """Grafica descomposición de componentes de Prophet."""
    futuro = pd.DataFrame({"ds": pd.to_datetime(df_train["fecha_t"].values)})
    fc = model.predict(futuro)
    ruta = DIR_OUTPUT / f"componentes_{banco}_fold{fold_num:02d}.png"
    try:
        fig = model.plot_components(fc)
        fig.suptitle(f"Componentes Prophet — {banco} fold {fold_num}", y=1.01)
        fig.savefig(ruta, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {ruta.name}")
    except Exception as e:
        print(f"  AVISO componentes fold {fold_num}: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not RUTA_MATRIZ.exists():
        raise FileNotFoundError(f"Matriz no encontrada: {RUTA_MATRIZ}")

    df_all = pd.read_parquet(RUTA_MATRIZ, filters=[("h", "==", H_REF)])
    df_all["fecha_t"] = pd.to_datetime(df_all["fecha_t"])
    bancos = BANCOS or df_all["banco"].unique().tolist()

    resumen = []

    for banco in bancos:
        print(f"\n{'═'*60}")
        print(f"  BANCO: {banco}")
        df = df_all[df_all["banco"] == banco].sort_values("fecha_t").reset_index(drop=True)

        for fold in FOLDS:
            fn = fold["fold"]
            print(f"\n  Fold {fn} | TRAIN→{fold['train_end']}  TEST: {fold['test_start']}→{fold['test_end']}")

            df_train = df[df["fecha_t"] <= pd.Timestamp(fold["train_end"])]
            df_val   = df[(df["fecha_t"] >= pd.Timestamp(fold["val_start"])) &
                          (df["fecha_t"] <= pd.Timestamp(fold["val_end"]))]
            df_test  = df[(df["fecha_t"] >= pd.Timestamp(fold["test_start"])) &
                          (df["fecha_t"] <= pd.Timestamp(fold["test_end"]))]

            if len(df_train) < 120:
                print(f"  Fold {fn}: train insuficiente ({len(df_train)} obs)")
                continue

            try:
                model = ajustar_prophet(df_train["fecha_t"].values,
                                        df_train["target"].values)
            except Exception as e:
                print(f"  Fold {fn}: Prophet falló — {e}")
                continue

            fc_val  = predecir(model, df_val["fecha_t"].values)
            fc_test = predecir(model, df_test["fecha_t"].values)

            # Métricas
            y_test   = df_test["target"].values
            y_val    = df_val["target"].values
            yhat_t   = fc_test["yhat"].values
            yhat_v   = fc_val["yhat"].values
            lower_t  = fc_test["yhat_lower"].values
            upper_t  = fc_test["yhat_upper"].values

            pb_q50_t = pinball(y_test, yhat_t, 0.50)
            pb_q50_v = pinball(y_val,  yhat_v, 0.50)
            cov_t    = cobertura(y_test, lower_t, upper_t)
            mae_t    = float(np.mean(np.abs(y_test - yhat_t)))

            print(f"    Pinball Q50 TEST={pb_q50_t/1e6:.1f}M  VAL={pb_q50_v/1e6:.1f}M  "
                  f"Cobertura80%={cov_t:.1%}  MAE={mae_t/1e6:.1f}M")

            resumen.append({
                "banco": banco, "fold": fn,
                "train_end": fold["train_end"],
                "test_start": fold["test_start"], "test_end": fold["test_end"],
                "pinball_q50_test_MM": round(pb_q50_t / 1e6, 2),
                "pinball_q50_val_MM":  round(pb_q50_v / 1e6, 2),
                "cobertura_80_test":   round(cov_t, 4),
                "mae_test_MM":         round(mae_t / 1e6, 2),
            })

            graficar_fold(fn, banco, df_train, df_test, df_val,
                          fc_test, fc_val, model)
            graficar_componentes(fn, banco, model, df_train)

    # Resumen CSV
    if resumen:
        df_res = pd.DataFrame(resumen)
        ruta_csv = DIR_OUTPUT / "resumen_prophet.csv"
        df_res.to_csv(ruta_csv, index=False)
        print(f"\n  Resumen guardado: {ruta_csv}")
        print(df_res.groupby("banco")[["pinball_q50_test_MM",
                                       "cobertura_80_test"]].mean().round(2))


if __name__ == "__main__":
    main()
