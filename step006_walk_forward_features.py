# -*- coding: utf-8 -*-
"""
step006_walk_forward_features.py
=================================
Walk-forward CV — Detección de estrategia sobreencaje BBVA.

Estructura:
  · Granularidad : mensual (una fila por mes)
  · Target       : estrategia (binaria: 1 = retiro > 50% saldo previo)
  · Ventana min  : 3 años de train (2020-2022)
  · Expansión    : +1 año por fold
  · Gap          : 1 mes entre fin de train y inicio de val (evita leakage de lags)

Folds:
  FOLD 1 : Train Ene-2020 → Dic-2022  |  Val Ene-2023 → Dic-2023
  FOLD 2 : Train Ene-2020 → Dic-2023  |  Val Ene-2024 → Dic-2024
  TEST   : Train Ene-2020 → Dic-2024  |  Test Ene-2025 → hoy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday, USFederalHolidayCalendar,
)
from pandas.tseries.offsets import Easter, Day as _Day

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

BASE    = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA    = BASE / "1. Data" / "Raw" / "bbva_encaje.xlsx"
RUTA_TX = BASE / "1. Data" / "Raw" / "Transacciones_BancaLocal.xlsx"
DIR_OUT = BASE / "2. Output" / "encaje_bbva"
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Parámetros estrategia
UMBRAL_SALDO   = 0.50   # retiro > 50% saldo → estrategia
UMBRAL_90      = 0.90   # umbral cumplimiento mensual
N_DIAS_HAB     = 6      # ventana días hábiles al cierre

# Walk-forward folds: (train_end, val_start, val_end)
INICIO_TRAIN = "2020-01"
FOLDS = [
    ("2022-12", "2023-02", "2023-12"),   # Fold 1 — gap 1 mes
    ("2023-12", "2024-02", "2024-12"),   # Fold 2 — gap 1 mes
]
TEST_START = "2025-02"   # gap 1 mes tras fin de train completo (2024-12)

# XGBoost
XGB_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = 3,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = 42,
    eval_metric       = "auc",
    verbosity         = 0,
)

# ══════════════════════════════════════════════════════════════════════════════
# CALENDARIO DÍAS HÁBILES
# ══════════════════════════════════════════════════════════════════════════════

class _PeruCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("AnioNuevo",   month=1,  day=1),
        Holiday("JuevesSanto", month=1,  day=1, offset=[Easter(), _Day(-3)]),
        GoodFriday,
        Holiday("Trabajo",     month=5,  day=1),
        Holiday("SanPedro",    month=6,  day=29),
        Holiday("FiestasP1",   month=7,  day=28),
        Holiday("FiestasP2",   month=7,  day=29),
        Holiday("SantaRosa",   month=8,  day=30),
        Holiday("Angamos",     month=10, day=8),
        Holiday("TodosSantos", month=11, day=1),
        Holiday("Inmaculada",  month=12, day=8),
        Holiday("Navidad",     month=12, day=25),
    ]

# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  Step 006 — Walk-Forward CV  |  Estrategia sobreencaje BBVA")
print("=" * 65)
print("\nCargando datos...")

df = pd.read_excel(RUTA, header=0).iloc[:, :9].copy()
df.columns = ["fecha", "entidad", "codigo", "overnight", "cta_cte",
               "caja", "tose", "exigible", "retiro_neto"]
df["fecha"]      = pd.to_datetime(df["fecha"])
df               = df.sort_values("fecha").reset_index(drop=True)
df["encaje"]     = df["cta_cte"] + df["caja"]
df["encaje_ovn"] = df["encaje"] + df["overnight"]

# Calendario
_f0 = df["fecha"].min()
_f1 = df["fecha"].max()
_hols = set(_PeruCalendar().holidays(_f0, _f1).normalize()) | \
        set(USFederalHolidayCalendar().holidays(_f0, _f1).normalize())
df["_is_bday"] = (df["fecha"].dt.weekday < 5) & (~df["fecha"].isin(_hols))
df["_am"]      = df["fecha"].dt.to_period("M")

# Acumulados mensuales (para capacidad_retiro)
df["EncajeAcumMes"]        = df.groupby("_am")["encaje"].cumsum()
df["NecAcumMes"]           = df.groupby("_am")["exigible"].cumsum()
df["dia_mes"]              = df["fecha"].dt.day
df["dias_en_mes"]          = df["fecha"].dt.days_in_month
df["ExigibleTotalMes_est"] = (df["NecAcumMes"] / df["dia_mes"]) * df["dias_en_mes"]

# Sistema
_sis_ok = False
try:
    _tx = pd.read_excel(RUTA_TX)
    _tx["fecha"] = pd.to_datetime(_tx["Fecha Valor"])
    _tx["monto"] = pd.to_numeric(_tx["Delivery Principal Usd"], errors="coerce")
    _flujo_sis   = _tx.groupby("fecha")["monto"].sum().rename("retiro_neto_sis").reset_index()
    df           = df.merge(_flujo_sis, on="fecha", how="left")
    _sis_ok      = True
    print(f"  Sistema cargado: {len(_flujo_sis):,} días")
except Exception as _e:
    df["retiro_neto_sis"] = np.nan
    print(f"  AVISO sistema: {_e}")

print(f"  Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"  Filas  : {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE FEATURES MENSUALES
# ══════════════════════════════════════════════════════════════════════════════

print("\nConstruyendo features mensuales...")

def _build_monthly(df):
    rows = []
    for _am, g in df.groupby("_am"):
        g      = g.sort_values("fecha")
        g_bday = g[g["_is_bday"]]
        late6  = g_bday.tail(N_DIAS_HAB)
        late15 = g_bday.tail(15)

        ret_6d  = late6["retiro_neto"].sum()  if len(late6)  else np.nan
        ret_15d = late15["retiro_neto"].sum() if len(late15) else np.nan
        med_sal = g["encaje_ovn"].median()
        max_sal = g["encaje_ovn"].max()
        umbral  = UMBRAL_SALDO * med_sal

        estrategia = (
            np.isfinite(ret_6d) and np.isfinite(umbral) and ret_6d < -umbral
        )

        # Concentración 6d vs 15d
        if (np.isfinite(ret_6d) and np.isfinite(ret_15d)
                and ret_15d < 0 and ret_6d < 0):
            concentracion = ret_6d / ret_15d
        else:
            concentracion = np.nan

        # Sistema
        sis_6d = (late6["retiro_neto_sis"].sum()
                  if "retiro_neto_sis" in late6.columns else np.nan)

        # Capacidad de retiro diaria:
        # exceso = EncajeAcumMes - ExigibleTotalMes_est × 0.90
        # cap    = exceso / n_dias_habiles_mes
        n_bday          = int(g_bday.shape[0])
        encaje_acum_mes = float(g["encaje"].sum())
        exig_total_est  = float(g["ExigibleTotalMes_est"].iloc[-1])
        exceso          = encaje_acum_mes - exig_total_est * UMBRAL_90
        cap_retiro      = exceso / n_bday if n_bday > 0 else np.nan

        rows.append({
            "_am":           _am,
            "anio":          _am.year,
            "mes_num":       _am.month,
            "fecha_plot":    _am.to_timestamp(),
            # ── target ──────────────────────────────────────────────────────
            "estrategia":    int(estrategia),
            # ── features retiro ─────────────────────────────────────────────
            "ret_6d_M":      ret_6d  / 1e6 if np.isfinite(ret_6d)  else np.nan,
            "ret_15d_M":     ret_15d / 1e6 if np.isfinite(ret_15d) else np.nan,
            "concentracion": round(concentracion, 4) if np.isfinite(concentracion) else np.nan,
            # ── features saldo ──────────────────────────────────────────────
            "med_sal_M":     med_sal / 1e6 if np.isfinite(med_sal) else np.nan,
            "max_sal_M":     max_sal / 1e6 if np.isfinite(max_sal) else np.nan,
            # ── capacidad de retiro ──────────────────────────────────────────
            "cap_retiro_M":  cap_retiro / 1e6 if np.isfinite(cap_retiro) else np.nan,
            "n_bday_mes":    n_bday,
            # ── sistema ─────────────────────────────────────────────────────
            "sis_ret_6d_M":  sis_6d / 1e6 if np.isfinite(sis_6d) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("fecha_plot").reset_index(drop=True)


dm = _build_monthly(df)

# ── Lag features (todos prospecivos: usan mes M-1 completo) ──────────────────
dm["estrategia_lag1"]    = dm["estrategia"].shift(1)
dm["ret_6d_lag1"]        = dm["ret_6d_M"].shift(1)
dm["max_sal_prev_M"]     = dm["max_sal_M"].shift(1)
dm["med_sal_prev_M"]     = dm["med_sal_M"].shift(1)

# retiro / máximo saldo mes previo (más conservador que mediana)
dm["ret_pct_max_prev"] = (
    -dm["ret_6d_M"] / dm["max_sal_prev_M"].replace(0, np.nan) * 100
).clip(lower=0)

# meses consecutivos con estrategia
_consec, _cnt = [], 0
for _e in dm["estrategia_lag1"].fillna(0):
    _cnt = _cnt + 1 if _e == 1 else 0
    _consec.append(_cnt)
dm["n_meses_consec"] = _consec

# share BBVA / Sistema en retiro
_mask = (dm["ret_6d_M"] < 0) & (dm["sis_ret_6d_M"] < 0)
dm["bbva_share_sis"] = np.where(
    _mask,
    dm["ret_6d_M"] / dm["sis_ret_6d_M"].replace(0, np.nan) * 100,
    np.nan,
)

# ── Dataset final ─────────────────────────────────────────────────────────────
FEATURES = [
    "ret_6d_M",          # 1. retiro acumulado últimos 6d hábiles
    "ret_15d_M",         # 2. retiro acumulado últimos 15d hábiles
    "concentracion",     # 3. concentración 6d vs 15d
    "cap_retiro_M",      # 4. capacidad retiro diaria (exceso / n_bday) ★
    "n_bday_mes",        # 5. total días hábiles en el mes
    "estrategia_lag1",   # 6. estrategia mes anterior
    "ret_6d_lag1",       # 7. retiro 6d mes anterior
    "max_sal_prev_M",    # 8. máximo saldo mes previo
    "ret_pct_max_prev",  # 9. retiro / máximo saldo previo
    "n_meses_consec",    # 10. meses consecutivos con estrategia
    "sis_ret_6d_M",      # 11. retiro sistema últimos 6d
    "bbva_share_sis",    # 12. BBVA / Sistema retiro 6d
    "mes_num",           # 13. mes del año (estacionalidad)
]
TARGET = "estrategia"

dm_model = (dm[dm["_am"] >= INICIO_TRAIN]
            .dropna(subset=["estrategia_lag1", "ret_6d_M", "cap_retiro_M"])
            .reset_index(drop=True))

n_pos = dm_model[TARGET].sum()
n_tot = len(dm_model)
print(f"  Meses disponibles (2020+) : {n_tot}")
print(f"  Estrategia activada       : {n_pos} / {n_tot} ({n_pos/n_tot*100:.0f}%)")

# Ajuste de clase para dataset desbalanceado
_scale_pos = (n_tot - n_pos) / max(n_pos, 1)
XGB_PARAMS["scale_pos_weight"] = round(_scale_pos, 2)

# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  Walk-Forward Cross-Validation")
print("=" * 65)

cv_results = []
fold_models = []

for fold_n, (train_end, val_start, val_end) in enumerate(FOLDS, 1):
    mask_tr = (dm_model["_am"] >= INICIO_TRAIN) & (dm_model["_am"] <= train_end)
    mask_va = (dm_model["_am"] >= val_start)    & (dm_model["_am"] <= val_end)

    X_tr = dm_model.loc[mask_tr, FEATURES].fillna(0)
    y_tr = dm_model.loc[mask_tr, TARGET]
    X_va = dm_model.loc[mask_va, FEATURES].fillna(0)
    y_va = dm_model.loc[mask_va, TARGET]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr)
    fold_models.append(model)

    prob_va = model.predict_proba(X_va)[:, 1]
    pred_va = (prob_va >= 0.5).astype(int)

    auc  = roc_auc_score(y_va, prob_va) if y_va.nunique() > 1 else np.nan
    prec = precision_score(y_va, pred_va, zero_division=0)
    rec  = recall_score(y_va, pred_va, zero_division=0)
    f1   = f1_score(y_va, pred_va, zero_division=0)

    print(f"\n  Fold {fold_n}: Train {INICIO_TRAIN}→{train_end}  "
          f"|  Val {val_start}→{val_end}")
    print(f"    N train : {mask_tr.sum()} meses   N val : {mask_va.sum()} meses")
    print(f"    AUC={auc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"    Positivos — real: {int(y_va.sum())}  "
          f"predichos: {int(pred_va.sum())}")
    cm = confusion_matrix(y_va, pred_va)
    print(f"    Confusion matrix:\n"
          f"      TN={cm[0,0]}  FP={cm[0,1]}\n"
          f"      FN={cm[1,0]}  TP={cm[1,1]}")

    cv_results.append({
        "fold": fold_n, "train_end": train_end,
        "val_start": val_start, "val_end": val_end,
        "n_train": int(mask_tr.sum()), "n_val": int(mask_va.sum()),
        "auc": round(auc, 4) if np.isfinite(auc) else np.nan,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": int(cm[1,1]), "fp": int(cm[0,1]),
        "tn": int(cm[0,0]), "fn": int(cm[1,0]),
    })

# ══════════════════════════════════════════════════════════════════════════════
# TEST FINAL
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  Test final")
print(f"  Train {INICIO_TRAIN}→2024-12  |  Test {TEST_START}→fin")
print(f"{'='*65}")

mask_tr_f = (dm_model["_am"] >= INICIO_TRAIN) & (dm_model["_am"] <= "2024-12")
mask_te   =  dm_model["_am"] >= TEST_START

X_tr_f = dm_model.loc[mask_tr_f, FEATURES].fillna(0)
y_tr_f = dm_model.loc[mask_tr_f, TARGET]
X_te   = dm_model.loc[mask_te,   FEATURES].fillna(0)
y_te   = dm_model.loc[mask_te,   TARGET]

model_final = XGBClassifier(**XGB_PARAMS)
model_final.fit(X_tr_f, y_tr_f)

prob_te = model_final.predict_proba(X_te)[:, 1]
pred_te = (prob_te >= 0.5).astype(int)

auc_te  = roc_auc_score(y_te, prob_te) if y_te.nunique() > 1 else np.nan
prec_te = precision_score(y_te, pred_te, zero_division=0)
rec_te  = recall_score(y_te, pred_te, zero_division=0)
f1_te   = f1_score(y_te, pred_te, zero_division=0)
cm_te   = confusion_matrix(y_te, pred_te)

print(f"  N train: {mask_tr_f.sum()} meses   N test: {mask_te.sum()} meses")
print(f"  AUC={auc_te:.3f}  Precision={prec_te:.3f}  "
      f"Recall={rec_te:.3f}  F1={f1_te:.3f}")
print(f"  Positivos — real: {int(y_te.sum())}  predichos: {int(pred_te.sum())}")
print(f"  Confusion matrix:\n"
      f"    TN={cm_te[0,0]}  FP={cm_te[0,1]}\n"
      f"    FN={cm_te[1,0]}  TP={cm_te[1,1]}")
print(f"\n  Reporte completo:")
print(classification_report(y_te, pred_te,
                             target_names=["Sin estrategia", "Estrategia"]))

cv_results.append({
    "fold": "TEST", "train_end": "2024-12",
    "val_start": TEST_START, "val_end": "fin",
    "n_train": int(mask_tr_f.sum()), "n_val": int(mask_te.sum()),
    "auc": round(auc_te, 4) if np.isfinite(auc_te) else np.nan,
    "precision": round(prec_te, 4), "recall": round(rec_te, 4),
    "f1": round(f1_te, 4),
    "tp": int(cm_te[1,1]), "fp": int(cm_te[0,1]),
    "tn": int(cm_te[0,0]), "fn": int(cm_te[1,0]),
})

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

fi = (pd.DataFrame({"feature": FEATURES,
                    "importance": model_final.feature_importances_})
      .sort_values("importance", ascending=False)
      .reset_index(drop=True))

print(f"\n{'='*65}")
print("  Feature Importance — modelo final (gain)")
print(f"{'='*65}")
for _, row in fi.iterrows():
    bar = "█" * int(row["importance"] * 60)
    print(f"  {row['feature']:<22}  {row['importance']:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

# 1 — Feature importance
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(fi["feature"][::-1], fi["importance"][::-1],
        color="#1565C0", alpha=0.82)
ax.set_xlabel("Importance (gain)", fontsize=10)
ax.set_title("Feature Importance — XGBoost modelo final\n"
             f"Train {INICIO_TRAIN}→2024-12  |  Test {TEST_START}→fin",
             fontweight="bold", fontsize=11)
plt.tight_layout()
fig.savefig(DIR_OUT / "06_feature_importance.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 2 — Probabilidad predicha vs real (test)
_fechas_te = dm_model.loc[mask_te, "fecha_plot"].values
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(_fechas_te, y_te.values,
       color="#90A4AE", alpha=0.55, width=25, label="Real (0/1)")
ax.plot(_fechas_te, prob_te,
        color="#E53935", lw=2, marker="o", ms=5, label="Prob. predicha")
ax.axhline(0.5, color="black", lw=1, ls="--", label="Umbral 0.5")
ax.set_ylabel("Probabilidad / Activación")
ax.set_title("Test — Probabilidad predicha vs estrategia real\n"
             f"Período {TEST_START} → fin",
             fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.15)
plt.tight_layout()
fig.savefig(DIR_OUT / "06_predicciones_test.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 3 — Métricas por fold
_df_cv = pd.DataFrame([r for r in cv_results if r["fold"] != "TEST"])
if not _df_cv.empty:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Walk-Forward CV — Métricas por fold", fontweight="bold")
    for ax, metric in zip(axes, ["auc", "precision", "recall"]):
        ax.bar(_df_cv["fold"].astype(str), _df_cv[metric],
               color="#1565C0", alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.upper())
        ax.set_xlabel("Fold")
        for i, v in enumerate(_df_cv[metric]):
            if np.isfinite(v):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(DIR_OUT / "06_metricas_cv.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

print(f"\n  Gráficos guardados:")
print(f"    · 06_feature_importance.png")
print(f"    · 06_predicciones_test.png")
print(f"    · 06_metricas_cv.png")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT EXCEL
# ══════════════════════════════════════════════════════════════════════════════

_ruta_xl = DIR_OUT / "06_walk_forward_results.xlsx"
with pd.ExcelWriter(_ruta_xl, engine="openpyxl") as _wr:

    # Hoja 1 — Resultados CV
    pd.DataFrame(cv_results).to_excel(_wr, sheet_name="CV_Resultados", index=False)

    # Hoja 2 — Feature importance
    fi.to_excel(_wr, sheet_name="Feature_Importance", index=False)

    # Hoja 3 — Predicciones test
    _pred_df = dm_model.loc[mask_te, ["_am", "fecha_plot", TARGET] + FEATURES].copy()
    _pred_df["prob_estrategia"] = prob_te
    _pred_df["pred_estrategia"] = pred_te
    _pred_df["correcto"]        = (_pred_df[TARGET] == _pred_df["pred_estrategia"]).astype(int)
    _pred_df.to_excel(_wr, sheet_name="Predicciones_Test", index=False)

    # Hoja 4 — Dataset completo con features
    dm_model[["_am", "fecha_plot", TARGET] + FEATURES].to_excel(
        _wr, sheet_name="Dataset_Completo", index=False)

    # Hoja 5 — Descripción de features
    _feat_desc = pd.DataFrame({
        "feature": FEATURES,
        "descripcion": [
            "Retiro acumulado últimos 6 días hábiles del mes (M USD)",
            "Retiro acumulado últimos 15 días hábiles del mes (M USD)",
            "Concentración: retiro_6d / retiro_15d (solo cuando ambos < 0)",
            "Capacidad retiro diaria: (EncajeAcum - Exigible×0.9) / n_bday (M USD)",
            "Total días hábiles en el mes",
            "Estrategia activada en mes anterior (0/1)",
            "Retiro 6d del mes anterior (M USD)",
            "Máximo saldo encaje OVN del mes anterior (M USD)",
            "Retiro / máximo saldo mes previo × 100 (%)",
            "Meses consecutivos con estrategia activada hasta mes anterior",
            "Retiro neto sistema últimos 6 días hábiles (M USD)",
            "BBVA / Sistema en retiro 6d (%)",
            "Mes del año (1-12)",
        ],
        "prospectivo": ["✅"] * len(FEATURES),
    })
    _feat_desc.to_excel(_wr, sheet_name="Features_Descripcion", index=False)

print(f"\n  Exportado: {_ruta_xl.name}")
print(f"  Hojas: CV_Resultados | Feature_Importance | Predicciones_Test "
      f"| Dataset_Completo | Features_Descripcion")

print(f"\n{'='*65}")
print(f"  Archivos en: {DIR_OUT}")
print(f"{'='*65}")
