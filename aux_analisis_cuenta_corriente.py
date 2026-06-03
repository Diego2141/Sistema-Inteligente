# -*- coding: utf-8 -*-
"""
aux_analisis_cuenta_corriente.py

Análisis exploratorio del saldo de Cuenta Corriente + OVN por banco en el BCRP.
Objetivo: evaluar si el nivel de CC y sus derivadas son features útiles para
predecir los retiros netos (D - R) en el modelo de step001/step005.

Preguntas que responde:
  1. ¿El perfil intramonth de CC confirma la hipótesis del encaje?
     (ingreso fuerte a inicio de mes, estabilidad, retiro de exceso a fin de mes)
  2. ¿El nivel de CC(t-1) tiene correlación con el flujo neto del día siguiente?
  3. ¿El exceso intramonth (CC vs. CC inicio de mes) predice el signo del flujo?
  4. ¿Los features derivados de CC tienen correlación incremental sobre los
     features estacionales ya presentes en la matriz?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

###############################################################################
# CONFIGURACIÓN
###############################################################################

RUTA_CC   = r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\Saldo fin del dia CC+OVN.xlsx"
RUTA_PARQUET = r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Clean\matriz_features.parquet"
DIR_OUT   = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\2. Output\analisis_cc")

# Banco a analizar en detalle. Debe coincidir con "Nombre Entidad" del Excel CC.
# Ejemplos visibles en el archivo: "BCP", "INTERBANK", "CITIBANK", "BBVA", etc.
# Para analizar el sistema completo, dejar como None → se agrega todos los bancos.
BANCO_FOCO = "BCP"
H_FOCO     = 2           # horizonte para correlaciones (h=2 = next business day)

DIR_OUT.mkdir(parents=True, exist_ok=True)

###############################################################################
# 1. LECTURA Y NORMALIZACIÓN DE LA CC
###############################################################################

print("=" * 65)
print("  ANÁLISIS DE CUENTA CORRIENTE + OVN — BCRP")
print("=" * 65)

print("\n[1] Leyendo archivo de CC...")
try:
    df_cc_raw = pd.read_excel(RUTA_CC, sheet_name="datos", header=0)
    print(f"    Filas: {len(df_cc_raw):,}  |  Columnas totales: {df_cc_raw.shape[1]}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"No se encontró el archivo:\n  {RUTA_CC}\n"
        "Verifica que el nombre y ruta sean correctos."
    )

# Estructura: col A = "Codigo Entidad", col B = "Nombre Entidad",
# col C en adelante = fechas (2018-01-01, 2018-01-02, ...)
# → convertir de formato wide a largo con melt

col_codigo = df_cc_raw.columns[0]   # "Codigo Entidad"
col_nombre = df_cc_raw.columns[1]   # "Nombre Entidad"
cols_fechas = df_cc_raw.columns[2:] # fechas como strings o datetime

df_cc = df_cc_raw.melt(
    id_vars=[col_codigo, col_nombre],
    value_vars=cols_fechas,
    var_name="fecha",
    value_name="cc",
)
df_cc = df_cc.rename(columns={col_nombre: "banco"})
df_cc["fecha"] = pd.to_datetime(df_cc["fecha"], errors="coerce")
df_cc["cc"]    = pd.to_numeric(df_cc["cc"], errors="coerce")
df_cc = df_cc.dropna(subset=["fecha", "cc"]).sort_values(["banco", "fecha"])

bancos_cc = sorted(df_cc["banco"].unique())
print(f"    Bancos encontrados ({len(bancos_cc)}): {bancos_cc}")
print(f"    Rango de fechas: {df_cc['fecha'].min().date()} → {df_cc['fecha'].max().date()}")

###############################################################################
# 2. FEATURES DERIVADOS DE LA CC
###############################################################################

print("\n[2] Calculando features derivados de CC...")

def calcular_features_cc(df_banco_cc: pd.DataFrame) -> pd.DataFrame:
    """
    Dado el DataFrame de CC de UN banco (columnas: fecha, cc),
    devuelve un DataFrame con los features derivados.
    """
    df = df_banco_cc.set_index("fecha").sort_index().copy()

    # Flujo neto implícito (D - R = ΔCC)
    df["flujo_neto_cc"]   = df["cc"].diff()

    # Lag 1 y 2 del saldo (estado antes del flujo)
    df["cc_lag1"]         = df["cc"].shift(1)
    df["cc_lag2"]         = df["cc"].shift(2)

    # Saldo al primer día hábil del mes
    df["periodo_mes"]     = df.index.to_period("M")
    df["cc_inicio_mes"]   = df.groupby("periodo_mes")["cc"].transform("first")

    # Ratio respecto al inicio de mes (posición relativa intramonth)
    df["cc_ratio_inicio"] = df["cc"] / df["cc_inicio_mes"].replace(0, np.nan)

    # Desviación respecto al promedio del mes hasta hoy
    df["cc_vs_prom_mes"]  = df["cc"] - df.groupby("periodo_mes")["cc"].transform(
        lambda x: x.expanding().mean()
    )

    # Acumulado neto del mes (equivalente a flujo_neto_acum_mes pero desde CC)
    df["flujo_acum_cc_mes"] = df.groupby("periodo_mes")["flujo_neto_cc"].transform(
        lambda x: x.fillna(0).cumsum()
    )

    # Volatilidad rolling 5d y 22d del saldo
    df["cc_vol_5d"]  = df["cc"].rolling(5).std()
    df["cc_vol_22d"] = df["cc"].rolling(22).std()

    return df.reset_index()

frames = []
for banco, grp in df_cc.groupby("banco"):
    feat = calcular_features_cc(grp[["fecha", "cc"]].copy())
    feat["banco"] = banco
    frames.append(feat)

df_feat = pd.concat(frames, ignore_index=True)
print(f"    Features calculados: {[c for c in df_feat.columns if c not in ['fecha','banco']]}")

###############################################################################
# 3. PERFIL INTRAMONTH DE CC — ¿SE CUMPLE LA HIPÓTESIS DEL ENCAJE?
###############################################################################

print("\n[3] Analizando perfil intramonth...")

df_foco = df_feat[df_feat["banco"] == BANCO_FOCO].copy() if BANCO_FOCO in df_feat["banco"].values else df_feat.copy()
if df_foco.empty:
    print(f"    Banco '{BANCO_FOCO}' no encontrado. Usando todos los bancos agregados.")
    df_foco = df_feat.groupby("fecha")[["cc", "flujo_neto_cc"]].sum().reset_index()
    df_foco["banco"] = "AGREGADO"

# Posición dentro del mes (día hábil 1, 2, 3, ...)
df_foco = df_foco.sort_values("fecha")
df_foco["periodo_mes"] = df_foco["fecha"].dt.to_period("M")
df_foco["pos_mes"] = df_foco.groupby("periodo_mes").cumcount() + 1

# Perfil promedio: CC normalizada (CC / CC inicio mes) por posición
perfil = df_foco.groupby("pos_mes").agg(
    cc_ratio_mean=("cc_ratio_inicio", "mean"),
    cc_ratio_p25 =("cc_ratio_inicio", lambda x: x.quantile(0.25)),
    cc_ratio_p75 =("cc_ratio_inicio", lambda x: x.quantile(0.75)),
    flujo_medio  =("flujo_neto_cc", "mean"),
    n_obs        =("cc", "count"),
).reset_index()
perfil = perfil[perfil["pos_mes"] <= 23]  # máx 23 días hábiles por mes

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(perfil["pos_mes"], perfil["cc_ratio_mean"], color="steelblue", lw=2, label="Media")
ax.fill_between(perfil["pos_mes"], perfil["cc_ratio_p25"], perfil["cc_ratio_p75"],
                alpha=0.25, color="steelblue", label="P25-P75")
ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_title(f"Perfil intramonth de CC\n{BANCO_FOCO} — CC(t) / CC(inicio_mes)", fontweight="bold")
ax.set_xlabel("Día hábil dentro del mes")
ax.set_ylabel("Ratio CC / CC inicio mes")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)

ax = axes[1]
colores = np.where(perfil["flujo_medio"] >= 0, "seagreen", "crimson")
ax.bar(perfil["pos_mes"], perfil["flujo_medio"] / 1e6, color=colores, alpha=0.75)
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Flujo neto medio por posición intramonth\n{BANCO_FOCO} (D − R = ΔCC)", fontweight="bold")
ax.set_xlabel("Día hábil dentro del mes")
ax.set_ylabel("Flujo neto medio (MM USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.25, axis="y")

plt.tight_layout()
ruta_fig1 = DIR_OUT / "01_perfil_intramonth_cc.png"
plt.savefig(ruta_fig1, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Guardado: {ruta_fig1.name}")

###############################################################################
# 4. CORRELACIÓN CC(t-1) → FLUJO NETO(t+1) y (t+2)
###############################################################################

print("\n[4] Correlaciones CC → flujo neto futuro...")

# Cargar parquet para cruzar con el flujo neto real del modelo
try:
    df_parquet = pd.read_parquet(RUTA_PARQUET)
    tiene_parquet = True
    print(f"    Parquet cargado: {df_parquet.shape}")
except Exception as e:
    tiene_parquet = False
    print(f"    Parquet no disponible ({e}) — usando flujo_neto_cc como proxy")

resultados_corr = []

for banco, grp in df_feat.groupby("banco"):
    grp = grp.sort_values("fecha").copy()

    if tiene_parquet:
        # Cruzar con flujo neto real del parquet (h=H_FOCO)
        pq_banco = df_parquet[
            (df_parquet["banco"] == banco) & (df_parquet["h"] == H_FOCO)
        ][["fecha_t", "target"]].rename(columns={"fecha_t": "fecha", "target": "flujo_real"})
        grp = grp.merge(pq_banco, on="fecha", how="left")
        col_flujo = "flujo_real"
    else:
        col_flujo = "flujo_neto_cc"

    grp["flujo_next1"] = grp[col_flujo].shift(-1)
    grp["flujo_next2"] = grp[col_flujo].shift(-2)

    for feat_col in ["cc", "cc_lag1", "cc_ratio_inicio", "cc_vs_prom_mes",
                     "flujo_acum_cc_mes", "cc_vol_5d"]:
        if feat_col not in grp.columns:
            continue
        sub = grp[[feat_col, "flujo_next1", "flujo_next2"]].dropna()
        if len(sub) < 30:
            continue
        r1 = sub[feat_col].corr(sub["flujo_next1"])
        r2 = sub[feat_col].corr(sub["flujo_next2"])
        resultados_corr.append({
            "banco": banco, "feature": feat_col,
            "corr_flujo_t+1": round(r1, 3),
            "corr_flujo_t+2": round(r2, 3),
        })

df_corr = pd.DataFrame(resultados_corr)
if not df_corr.empty:
    print("\n    Correlaciones (Pearson) con flujo neto futuro:")
    print(df_corr.to_string(index=False))

###############################################################################
# 5. EVOLUCIÓN TEMPORAL DE CC Y FLUJO NETO
###############################################################################

print("\n[5] Graficando evolución temporal...")

df_plot = df_feat[df_feat["banco"] == BANCO_FOCO].sort_values("fecha") if BANCO_FOCO in df_feat["banco"].values \
          else df_feat.groupby("fecha")[["cc", "flujo_neto_cc", "cc_ratio_inicio", "flujo_acum_cc_mes"]].sum().reset_index()

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax = axes[0]
ax.plot(df_plot["fecha"], df_plot["cc"] / 1e6, color="steelblue", lw=1.2)
ax.set_ylabel("Saldo CC (MM USD)")
ax.set_title(f"Cuenta Corriente en BCRP — {BANCO_FOCO}", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.2)

ax = axes[1]
colores_flujo = ["seagreen" if v >= 0 else "crimson"
                 for v in df_plot["flujo_neto_cc"].fillna(0)]
ax.bar(df_plot["fecha"], df_plot["flujo_neto_cc"].fillna(0) / 1e6,
       color=colores_flujo, alpha=0.7, width=1)
ax.axhline(0, color="black", lw=0.7)
ax.set_ylabel("ΔCC = D − R (MM USD)")
ax.set_title("Flujo neto implícito en CC (equivale al target del modelo)", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.2, axis="y")

ax = axes[2]
ax.plot(df_plot["fecha"], df_plot["flujo_acum_cc_mes"].fillna(0) / 1e6,
        color="darkorange", lw=1.2)
ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
ax.set_ylabel("Flujo acumulado del mes (MM USD)")
ax.set_title("Acumulado intramonth de ΔCC — proxy de exceso de encaje", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.2)

plt.tight_layout()
ruta_fig2 = DIR_OUT / "02_evolucion_temporal_cc.png"
plt.savefig(ruta_fig2, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Guardado: {ruta_fig2.name}")

###############################################################################
# 6. EXPORTAR FEATURES A EXCEL
###############################################################################

print("\n[6] Exportando features a Excel...")

ruta_excel = DIR_OUT / "features_cc_derivados.xlsx"
with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
    df_feat.to_excel(writer, index=False, sheet_name="Features_CC")
    perfil.to_excel(writer, index=False, sheet_name="Perfil_intramonth")
    if not df_corr.empty:
        df_corr.to_excel(writer, index=False, sheet_name="Correlaciones")

print(f"    Guardado: {ruta_excel.name}")

###############################################################################
# 7. CONCLUSIÓN: ¿VALE LA PENA AGREGAR A step001?
###############################################################################

print(f"\n{'='*65}")
print("  RESUMEN — ¿Agregar features de CC a la matriz?")
print(f"{'='*65}")

if not df_corr.empty:
    max_corr = df_corr[["corr_flujo_t+1", "corr_flujo_t+2"]].abs().max().max()
    umbral   = 0.10

    print(f"\n  Correlación máxima encontrada: {max_corr:.3f}")
    if max_corr >= umbral:
        print(f"  → RECOMENDACIÓN: SÍ agregar a step001.")
        print(f"    Los features de CC tienen correlación >{umbral:.0%} con el flujo futuro.")
        print(f"    Features candidatos:")
        top = df_corr[df_corr[["corr_flujo_t+1","corr_flujo_t+2"]].abs().max(axis=1) >= umbral]
        for _, r in top.iterrows():
            print(f"      · {r['feature']:<25} corr t+1={r['corr_flujo_t+1']:+.3f}  t+2={r['corr_flujo_t+2']:+.3f}")
    else:
        print(f"  → RECOMENDACIÓN: EVALUAR. Correlación lineal baja (<{umbral:.0%}).")
        print(f"    XGBoost puede capturar relaciones no lineales — revisar los plots.")
else:
    print("\n  No se pudo calcular correlaciones. Revisa los plots generados.")

print(f"\n  Plots generados en: {DIR_OUT}")
print(f"  · 01_perfil_intramonth_cc.png  — hipótesis del encaje")
print(f"  · 02_evolucion_temporal_cc.png — serie histórica + flujo + acumulado")
print(f"  · features_cc_derivados.xlsx   — tabla completa de features")
print()
