# -*- coding: utf-8 -*-
"""
aux_diagnostico_sistema.py
Diagnóstico exhaustivo del modelo SISTEMA antes de pasar a predicción.

Cubre 7 dimensiones:
  D1. NaN por feature (tasa real y patrón temporal)
  D2. Distribución del target (fat tails, skew, eventos extremos)
  D3. Series R_t0 y D_t0 del sistema (¿varían o están imputadas?)
  D4. Shift distribucional TRAIN → VAL → TEST (¿el mundo cambió?)
  D5. Calibración de las bandas (¿el Q95 realmente captura el 95%?)
  D6. Target por horizonte h (¿la incertidumbre crece con h?)
  D7. Importancia real vs ruido (¿los rezagos bancarios agregan valor?)

Salida: consola + 7 figuras PNG en 2. Output/diagnostico_sistema/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos"
DIR_OUT      = BASE_SISTEMA / "2. Output" / "diagnostico_sistema"
DIR_OUT.mkdir(parents=True, exist_ok=True)

BANCO        = "SISTEMA"
SEMANAS_VAL  = 26
SEMANAS_TEST = 13
COLS_EXCLUIR = {"fecha_t", "banco", "target"}

fmt_m = mticker.FuncFormatter(lambda x, _: f"{x/1e6:,.0f}")   # MM USD
fmt_k = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def cargar_sistema():
    print("Cargando SISTEMA desde Parquet...")
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", BANCO)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)
    print(f"  {len(df):,} filas | {df['fecha_t'].nunique():,} fechas | "
          f"target no-NaN: {df['target'].notna().sum():,}")
    return df


def split_tres(df):
    fechas = np.sort(df["fecha_t"].unique())
    n = len(fechas)
    n_test = min(SEMANAS_TEST * 5, n // 6)
    n_val  = min(SEMANAS_VAL  * 5, n // 4)
    if n - n_test - n_val < n // 2:
        n_val  = max(10, n // 6)
        n_test = max(5,  n // 8)
    c_val  = fechas[n - n_test - n_val]
    c_test = fechas[n - n_test]
    tr = df[df["fecha_t"] <  c_val].copy()
    va = df[(df["fecha_t"] >= c_val) & (df["fecha_t"] < c_test)].copy()
    te = df[df["fecha_t"] >= c_test].copy()
    return tr, va, te


def get_feat_cols(df):
    return [c for c in df.columns
            if c not in COLS_EXCLUIR and df[c].dtype.kind in ("f","i","u","b")]


def guardar(fig, nombre):
    ruta = DIR_OUT / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
# D1. NaN por feature
# ─────────────────────────────────────────────────────────────────────────────
def diag_nan(df, df_tr):
    print("\n[D1] NaN por feature...")
    cols = get_feat_cols(df)
    medianas_tr = df_tr[cols].median()

    nan_pct  = df[cols].isna().mean().sort_values(ascending=False)
    nan_full = nan_pct[nan_pct > 0]

    # Tasa de NaN antes y después de imputar con mediana de train
    cero_var = []
    for c in cols:
        imputado = df[c].fillna(medianas_tr[c])
        if imputado.std() < 1e-9:
            cero_var.append(c)

    print(f"  Features con NaN   : {len(nan_full)} / {len(cols)}")
    print(f"  Features cero-var  : {len(cero_var)}")
    if cero_var:
        print(f"    {cero_var}")
    print("\n  Top 20 NaN (%):")
    print(nan_full.head(20).mul(100).round(1).to_string())

    # Figura: barras de NaN %
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    if len(nan_full) > 0:
        nan_full.head(30).mul(100).plot.barh(ax=ax, color="coral", alpha=0.85)
        ax.set_xlabel("% NaN", fontsize=10)
        ax.set_title("Features con NaN (top 30)", fontweight="bold")
        ax.axvline(5,  color="orange", lw=1, ls="--", label="5%")
        ax.axvline(20, color="red",    lw=1, ls="--", label="20%")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Sin NaN", ha="center", va="center", fontsize=14)
        ax.set_title("Features con NaN")

    # Evolución temporal de NaN en features bancarias clave
    ax2 = axes[1]
    cols_banco = [c for c in ["R_t0","D_t0","R_t-1","D_t-1","sigma_R_5d","ma_R_22d"]
                  if c in df.columns]
    if cols_banco:
        nan_mes = (df.set_index("fecha_t")[cols_banco]
                     .resample("ME").apply(lambda x: x.isna().mean())
                     .rename_axis("fecha"))
        for c in cols_banco:
            ax2.plot(nan_mes.index, nan_mes[c]*100, label=c, alpha=0.8)
        ax2.set_ylabel("% NaN", fontsize=10)
        ax2.set_title("Evolución temporal de NaN (features bancarias)", fontweight="bold")
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("D1 — NaN por Feature — SISTEMA", fontweight="bold", fontsize=13)
    plt.tight_layout()
    guardar(fig, "D1_nan_features.png")
    return nan_full, cero_var


# ─────────────────────────────────────────────────────────────────────────────
# D2. Distribución del target
# ─────────────────────────────────────────────────────────────────────────────
def diag_target(df, df_tr, df_va, df_te):
    print("\n[D2] Distribución del target...")
    # Solo h_min para evitar correlación por horizonte
    h_min = int(df["h"].min())
    tgt_tr = df_tr.loc[df_tr["h"]==h_min, "target"].dropna() / 1e6
    tgt_va = df_va.loc[df_va["h"]==h_min, "target"].dropna() / 1e6
    tgt_te = df_te.loc[df_te["h"]==h_min, "target"].dropna() / 1e6
    tgt_all = df.loc[df["h"]==h_min, "target"].dropna() / 1e6

    for nombre, s in [("TRAIN",tgt_tr),("VAL",tgt_va),("TEST",tgt_te)]:
        p99, p01 = s.quantile(0.99), s.quantile(0.01)
        print(f"  {nombre:5s}: media={s.mean():+.0f}  std={s.std():.0f}  "
              f"skew={s.skew():.2f}  kurt={s.kurtosis():.1f}  "
              f"P01={p01:.0f}  P99={p99:.0f}  "
              f"|x|>500MM: {(s.abs()>500).mean():.1%}")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # Histograma comparativo
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.linspace(tgt_all.quantile(0.005), tgt_all.quantile(0.995), 80)
    for s, lbl, col in [(tgt_tr,"TRAIN","steelblue"),(tgt_va,"VAL","orange"),(tgt_te,"TEST","crimson")]:
        ax1.hist(s, bins=bins, alpha=0.45, density=True, label=lbl, color=col)
    ax1.set_xlabel("Flujo neto D−R (MM USD) — solo h mínimo", fontsize=10)
    ax1.set_ylabel("Densidad", fontsize=10)
    ax1.set_title("Distribución del target: TRAIN vs VAL vs TEST", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Serie temporal del target (h mínimo)
    ax2 = fig.add_subplot(gs[1, :])
    for split_df, lbl, col in [(df_tr,"TRAIN","steelblue"),(df_va,"VAL","orange"),(df_te,"TEST","crimson")]:
        sub = split_df[split_df["h"]==h_min].set_index("fecha_t")["target"].dropna() / 1e6
        ax2.plot(sub.index, sub.values, color=col, lw=0.9, alpha=0.85, label=lbl)
    ax2.axhline(0, color="black", lw=0.6, ls="--")
    ax2.set_ylabel("MM USD", fontsize=10)
    ax2.set_title(f"Target en el tiempo (h={h_min})", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
    ax2.yaxis.set_major_formatter(fmt_k)

    fig.suptitle("D2 — Distribución del Target — SISTEMA", fontweight="bold", fontsize=13)
    guardar(fig, "D2_distribucion_target.png")


# ─────────────────────────────────────────────────────────────────────────────
# D3. Series R_t0 y D_t0 — ¿varían o están imputadas?
# ─────────────────────────────────────────────────────────────────────────────
def diag_series_bancarias(df, df_tr):
    print("\n[D3] Series bancarias R_t0 / D_t0...")
    cols_check = [c for c in ["R_t0","D_t0","R_t-1","D_t-1"] if c in df.columns]
    if not cols_check:
        print("  ALERTA: R_t0 y D_t0 no están en la matriz.")
        return

    medianas = df_tr[cols_check].median()
    h_min = int(df["h"].min())
    sub = df[df["h"]==h_min].set_index("fecha_t")[cols_check].sort_index()

    for c in cols_check:
        n_nan    = sub[c].isna().sum()
        pct_med  = (sub[c].dropna() == medianas[c]).mean()
        std_real = sub[c].dropna().std()
        print(f"  {c:12s}: NaN={n_nan} ({n_nan/len(sub):.1%})  "
              f"std={std_real:,.0f}  "
              f"pct_igual_mediana={pct_med:.1%}")

    fig, axes = plt.subplots(len(cols_check), 1,
                             figsize=(15, 3*len(cols_check)), sharex=True)
    if len(cols_check) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols_check):
        ax.plot(sub.index, sub[c]/1e6, lw=0.8, color="steelblue", alpha=0.85)
        med_val = medianas[c] / 1e6
        ax.axhline(med_val, color="red", lw=1, ls="--",
                   label=f"Mediana imputación: {med_val:,.0f} MM")
        nan_mask = sub[c].isna()
        if nan_mask.any():
            ax.fill_between(sub.index, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=nan_mask, alpha=0.15, color="red", label="NaN imputado")
        ax.set_ylabel("MM USD", fontsize=9)
        ax.set_title(c, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(fmt_k)

    fig.suptitle("D3 — Series Bancarias del SISTEMA (h mínimo)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    guardar(fig, "D3_series_bancarias.png")


# ─────────────────────────────────────────────────────────────────────────────
# D4. Shift distribucional TRAIN → VAL → TEST
# ─────────────────────────────────────────────────────────────────────────────
def diag_shift(df, df_tr, df_va, df_te):
    print("\n[D4] Shift distribucional entre splits...")
    cols_feat = get_feat_cols(df)
    cols_feat = [c for c in cols_feat if c != "target"]

    stats = {}
    for nombre, split in [("TRAIN",df_tr),("VAL",df_va),("TEST",df_te)]:
        s = split[cols_feat].describe().loc[["mean","std"]].T
        s.columns = [f"{nombre}_mean", f"{nombre}_std"]
        stats[nombre] = s

    resumen = pd.concat([stats["TRAIN"], stats["VAL"], stats["TEST"]], axis=1)
    # Cambio relativo media: (VAL-TRAIN)/|TRAIN|
    eps = 1e-9
    resumen["shift_val_pct"]  = ((resumen["VAL_mean"]  - resumen["TRAIN_mean"]) /
                                  (resumen["TRAIN_mean"].abs() + eps) * 100)
    resumen["shift_test_pct"] = ((resumen["TEST_mean"] - resumen["TRAIN_mean"]) /
                                  (resumen["TRAIN_mean"].abs() + eps) * 100)

    top_shift = resumen["shift_test_pct"].abs().sort_values(ascending=False).head(15)
    print("  Top 15 features con mayor shift media TRAIN→TEST (%):")
    print(top_shift.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    top_feat = resumen["shift_test_pct"].abs().sort_values(ascending=False).head(20)
    vals = resumen.loc[top_feat.index, "shift_test_pct"]
    colors = ["crimson" if v > 0 else "steelblue" for v in vals]
    ax.barh(top_feat.index, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Cambio relativo media TRAIN→TEST (%)", fontsize=10)
    ax.set_title("Shift distribucional por feature (top 20)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    ax2 = axes[1]
    top_feat_val = resumen["shift_val_pct"].abs().sort_values(ascending=False).head(20)
    vals_val = resumen.loc[top_feat_val.index, "shift_val_pct"]
    colors_val = ["crimson" if v > 0 else "steelblue" for v in vals_val]
    ax2.barh(top_feat_val.index, vals_val, color=colors_val, alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Cambio relativo media TRAIN→VAL (%)", fontsize=10)
    ax2.set_title("Shift distribucional por feature (TRAIN→VAL)", fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("D4 — Shift Distribucional entre Splits — SISTEMA",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    guardar(fig, "D4_shift_distribucional.png")


# ─────────────────────────────────────────────────────────────────────────────
# D5. Calibración de las bandas
# ─────────────────────────────────────────────────────────────────────────────
def diag_calibracion(df_te, dir_modelos):
    print("\n[D5] Calibración de bandas en TEST...")

    import glob, json
    from datetime import date

    # Buscar el modelo más reciente de SISTEMA
    archivos = sorted(glob.glob(str(dir_modelos / f"lgbm_{BANCO}_q50_*.txt")))
    if not archivos:
        print("  No se encontraron modelos de SISTEMA — ejecutar step003 primero.")
        return

    # Cargar metadata para obtener las features en el orden correcto
    meta_archivos = sorted(glob.glob(str(dir_modelos / f"metadata_{BANCO}_*.json")))
    if not meta_archivos:
        print("  No se encontró metadata de SISTEMA.")
        return

    with open(meta_archivos[-1]) as f:
        meta = json.load(f)
    cols_feat = meta["features"]
    medianas  = df_te[cols_feat].median()  # aproximación: usar mediana del test

    df_eval = df_te[df_te["target"].notna()].copy()
    df_eval[cols_feat] = df_eval[cols_feat].fillna(medianas)
    X_te = df_eval[cols_feat]
    y_te = df_eval["target"].values / 1e6

    # Cargar los 5 modelos
    quantiles  = [0.01, 0.05, 0.50, 0.95, 0.99]
    preds_raw  = {}
    for tau in quantiles:
        archivos_q = sorted(glob.glob(
            str(dir_modelos / f"lgbm_{BANCO}_q{int(tau*100):02d}_*.txt")
        ))
        if not archivos_q:
            print(f"  Modelo q{int(tau*100):02d} no encontrado.")
            return
        booster = lgb.Booster(model_file=archivos_q[-1])
        preds_raw[tau] = booster.predict(X_te) / 1e6

    # Corrección cruce
    matrix = np.sort(np.column_stack([preds_raw[t] for t in quantiles]), axis=1)
    preds  = {t: matrix[:, i] for i, t in enumerate(quantiles)}

    # Cobertura por par de cuantiles
    pares = [(0.01, 0.99, "Q01–Q99", 98),
             (0.05, 0.95, "Q05–Q95", 90),
             (0.50, 0.50, "Q50 (mediana)", None)]

    print(f"\n  {'Banda':<14} {'Cobertura esperada':>20} {'Cobertura real':>16} {'Sesgo':>8}")
    print(f"  {'-'*62}")
    coberturas = {}
    for tau_lo, tau_hi, lbl, esperada in pares:
        if tau_lo == tau_hi:
            # RMSE y sesgo de mediana
            err = y_te - preds[0.50]
            rmse = np.sqrt(np.mean(err**2))
            sesgo = np.mean(err)
            print(f"  {lbl:<14} {'—':>20} {'RMSE':>10} {rmse:>8.1f}  sesgo={sesgo:+.1f}")
        else:
            dentro = ((y_te >= preds[tau_lo]) & (y_te <= preds[tau_hi])).mean()
            coberturas[lbl] = dentro
            flag = "✓" if abs(dentro - esperada/100) < 0.05 else "⚠"
            print(f"  {lbl:<14} {esperada:>19}%  {dentro:>14.1%}  {flag}")

    # Figura: cobertura por horizonte h
    hs = sorted(df_eval["h"].unique())
    cob_h_95 = []
    cob_h_99 = []
    for h in hs:
        mask = df_eval["h"] == h
        idx  = np.where(mask.values)[0]
        y_h  = y_te[idx]
        dentro_95 = ((y_h >= preds[0.05][idx]) & (y_h <= preds[0.95][idx])).mean()
        dentro_99 = ((y_h >= preds[0.01][idx]) & (y_h <= preds[0.99][idx])).mean()
        cob_h_95.append(dentro_95)
        cob_h_99.append(dentro_99)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(hs, [c*100 for c in cob_h_95], color="steelblue", lw=1.5, label="Q05–Q95")
    ax.plot(hs, [c*100 for c in cob_h_99], color="navy", lw=1.5, label="Q01–Q99")
    ax.axhline(90, color="steelblue", lw=1, ls="--", alpha=0.6, label="90% esperado")
    ax.axhline(98, color="navy",      lw=1, ls="--", alpha=0.6, label="98% esperado")
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=10)
    ax.set_ylabel("Cobertura real (%)", fontsize=10)
    ax.set_title("Calibración por horizonte h (TEST)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Scatter: realizado vs mediana predicha (muestra aleatoria de 1000 obs)
    ax2 = axes[1]
    n_muestra = min(1000, len(y_te))
    idx_m = np.random.choice(len(y_te), n_muestra, replace=False)
    ax2.scatter(preds[0.50][idx_m], y_te[idx_m], alpha=0.25, s=8, color="steelblue")
    lim = max(abs(y_te).max(), abs(preds[0.50]).max()) * 1.05
    ax2.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="45°")
    ax2.set_xlabel("Mediana predicha (MM USD)", fontsize=10)
    ax2.set_ylabel("Realizado (MM USD)", fontsize=10)
    ax2.set_title("Realizado vs Mediana predicha (TEST, muestra)", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle("D5 — Calibración de Bandas — SISTEMA", fontweight="bold", fontsize=13)
    plt.tight_layout()
    guardar(fig, "D5_calibracion.png")


# ─────────────────────────────────────────────────────────────────────────────
# D6. Target por horizonte h
# ─────────────────────────────────────────────────────────────────────────────
def diag_target_por_h(df, df_tr):
    print("\n[D6] Target por horizonte h...")
    tgt_h = df_tr.groupby("h")["target"].agg(
        media="mean", std="std",
        p05=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
        p01=lambda x: x.quantile(0.01),
        p99=lambda x: x.quantile(0.99),
    ) / 1e6
    hs = tgt_h.index

    print(f"  h={hs.min()}: std={tgt_h.loc[hs.min(),'std']:.0f} MM  "
          f"p01={tgt_h.loc[hs.min(),'p01']:.0f}  p99={tgt_h.loc[hs.min(),'p99']:.0f}")
    print(f"  h={hs.max()}: std={tgt_h.loc[hs.max(),'std']:.0f} MM  "
          f"p01={tgt_h.loc[hs.max(),'p01']:.0f}  p99={tgt_h.loc[hs.max(),'p99']:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.fill_between(hs, tgt_h["p01"], tgt_h["p99"], alpha=0.15, color="steelblue", label="P01–P99")
    ax.fill_between(hs, tgt_h["p05"], tgt_h["p95"], alpha=0.30, color="steelblue", label="P05–P95")
    ax.plot(hs, tgt_h["media"], color="steelblue", lw=1.8, label="Media")
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Horizonte h (días hábiles)", fontsize=10)
    ax.set_ylabel("MM USD", fontsize=10)
    ax.set_title("Distribución del target por h (TRAIN)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(fmt_k)

    ax2 = axes[1]
    ax2.plot(hs, tgt_h["std"], color="crimson", lw=1.8)
    ax2.set_xlabel("Horizonte h (días hábiles)", fontsize=10)
    ax2.set_ylabel("Desviación estándar (MM USD)", fontsize=10)
    ax2.set_title("Incertidumbre (std) del target por h (TRAIN)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(fmt_k)

    fig.suptitle("D6 — Target por Horizonte h — SISTEMA", fontweight="bold", fontsize=13)
    plt.tight_layout()
    guardar(fig, "D6_target_por_h.png")


# ─────────────────────────────────────────────────────────────────────────────
# D7. Valor predictivo real de los rezagos bancarios
# ─────────────────────────────────────────────────────────────────────────────
def diag_correlacion_rezagos(df, df_tr):
    print("\n[D7] Correlación rezagos bancarios vs target...")
    h_min = int(df["h"].min())
    sub = df_tr[df_tr["h"] == h_min].copy()

    cols_banco = [c for c in sub.columns
                  if any(c.startswith(p) for p in
                         ["R_t","D_t","sigma_R","sigma_D","ma_R","ma_D","delta_R","delta_D",
                          "R_conf","D_conf"])
                  and c in get_feat_cols(sub)]

    if not cols_banco:
        print("  No se encontraron features bancarias.")
        return

    medianas = df_tr[cols_banco].median()
    sub[cols_banco] = sub[cols_banco].fillna(medianas)

    corr = sub[cols_banco + ["target"]].corr()["target"].drop("target")
    corr_abs = corr.abs().sort_values(ascending=False)

    print("  Correlación |r| con target (h mínimo):")
    print(corr_abs.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    colors = ["crimson" if v < 0 else "steelblue" for v in corr.loc[corr_abs.index]]
    corr.loc[corr_abs.index].plot.barh(ax=ax, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Correlación de Pearson con target", fontsize=10)
    ax.set_title(f"Correlación features bancarias vs target (h={h_min})", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Scatter R_t0 vs target (el rezago más importante en teoría)
    ax2 = axes[1]
    if "R_t0" in sub.columns and "target" in sub.columns:
        x = sub["R_t0"].dropna() / 1e6
        y = sub.loc[x.index, "target"] / 1e6
        ax2.scatter(x, y, alpha=0.15, s=6, color="steelblue")
        # Línea de tendencia
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax2.plot(xr, m*xr + b, "r--", lw=1.5, label=f"r={x.corr(y):.3f}")
        ax2.set_xlabel("R_t0 sistema (MM USD)", fontsize=10)
        ax2.set_ylabel("Target D−R (MM USD)", fontsize=10)
        ax2.set_title(f"R_t0 vs Target (h={h_min}, TRAIN)", fontweight="bold")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(fmt_k)
        ax2.yaxis.set_major_formatter(fmt_k)

    fig.suptitle("D7 — Valor Predictivo de Rezagos Bancarios — SISTEMA",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    guardar(fig, "D7_correlacion_rezagos.png")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
def resumen_ejecutivo(df, df_tr, df_va, df_te, nan_full, cero_var):
    print("\n" + "="*70)
    print("RESUMEN EJECUTIVO — DIAGNÓSTICO SISTEMA")
    print("="*70)

    h_min = int(df["h"].min())
    tgt_tr = df_tr.loc[df_tr["h"]==h_min, "target"].dropna() / 1e6
    tgt_te = df_te.loc[df_te["h"]==h_min, "target"].dropna() / 1e6

    print(f"\n  Período TRAIN : {df_tr['fecha_t'].min().date()} → {df_tr['fecha_t'].max().date()}")
    print(f"  Período VAL   : {df_va['fecha_t'].min().date()} → {df_va['fecha_t'].max().date()}")
    print(f"  Período TEST  : {df_te['fecha_t'].min().date()} → {df_te['fecha_t'].max().date()}")
    print(f"\n  std target TRAIN : {tgt_tr.std():,.0f} MM USD")
    print(f"  std target TEST  : {tgt_te.std():,.0f} MM USD  "
          f"(ratio: {tgt_te.std()/tgt_tr.std():.2f}x)")
    print(f"\n  Features con NaN  : {len(nan_full)}")
    print(f"  Features cero-var : {len(cero_var)}")

    # Señales de alerta
    alertas = []
    if tgt_te.std() / tgt_tr.std() > 1.5:
        alertas.append("⚠ TEST tiene volatilidad >1.5× el TRAIN → regime shift")
    if len(cero_var) > 0:
        alertas.append(f"⚠ {len(cero_var)} feature(s) con varianza ~0 tras imputación")
    if len(nan_full) > 10:
        alertas.append(f"⚠ {len(nan_full)} features con NaN → imputación puede sesgar")
    if nan_full.max() > 0.30 if len(nan_full) > 0 else False:
        alertas.append(f"⚠ Hay features con >30% NaN: {nan_full[nan_full>0.30].index.tolist()}")

    if alertas:
        print("\n  ALERTAS:")
        for a in alertas:
            print(f"    {a}")
    else:
        print("\n  Sin alertas críticas detectadas.")

    print(f"\n  Gráficos guardados en: {DIR_OUT}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("DIAGNÓSTICO EXHAUSTIVO — SISTEMA")
    print("="*70)

    df = cargar_sistema()
    df_tr, df_va, df_te = split_tres(df)

    nan_full, cero_var = diag_nan(df, df_tr)
    diag_target(df, df_tr, df_va, df_te)
    diag_series_bancarias(df, df_tr)
    diag_shift(df, df_tr, df_va, df_te)
    diag_calibracion(df_te, DIR_MODELOS)
    diag_target_por_h(df, df_tr)
    diag_correlacion_rezagos(df, df_tr)
    resumen_ejecutivo(df, df_tr, df_va, df_te, nan_full, cero_var)
