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

# h=1 y h=2 tienen datos confirmados (R_conf, D_conf) → no aportan señal incremental.
# El interés real de la CC está en h=3 en adelante.
H_EVALUAR = [3, 5, 10, 22, 45, 90]

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

    # Acumulado rolling (CC(t) - CC(t-N)): variación neta sin depender del mes
    # Positivo → banco ha acumulado → probable retiro próximo
    # Negativo → banco ha retirado → probable depósito próximo
    for n in [5, 10, 22]:
        df[f"flujo_acum_roll_{n}d"] = df["cc"] - df["cc"].shift(n)

    return df.reset_index()

frames = []
for banco, grp in df_cc.groupby("banco"):
    feat = calcular_features_cc(grp[["fecha", "cc"]].copy())
    feat["banco"] = banco
    frames.append(feat)

df_feat = pd.concat(frames, ignore_index=True)
print(f"    Features calculados: {[c for c in df_feat.columns if c not in ['fecha','banco']]}")

# ── Agregar SISTEMA: suma de todas las entidades ──────────────────────────────
# "SISTEMA" en el parquet = agregado de los 63 bancos → sumar CC de todos ellos
print("\n[2b] Construyendo CC_SISTEMA (suma de las 63 entidades)...")
df_cc_sistema = (
    df_cc.groupby("fecha")["cc"]
    .sum()
    .reset_index()
)
print(f"     Rango: {df_cc_sistema['fecha'].min().date()} → "
      f"{df_cc_sistema['fecha'].max().date()}  |  {len(df_cc_sistema):,} obs")

feat_sistema = calcular_features_cc(df_cc_sistema[["fecha", "cc"]].copy())
feat_sistema["banco"] = "SISTEMA"
df_feat = pd.concat([df_feat, feat_sistema], ignore_index=True)

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

FEATURES_CC = ["cc", "cc_lag1", "cc_vs_prom_mes", "cc_ratio_inicio",
               "flujo_acum_cc_mes", "cc_vol_5d",
               "flujo_acum_roll_5d", "flujo_acum_roll_10d", "flujo_acum_roll_22d"]

resultados_corr = []

for banco, grp in df_feat.groupby("banco"):
    grp = grp.sort_values("fecha").copy()

    for h in H_EVALUAR:
        if not tiene_parquet:
            grp[f"flujo_h{h}"] = grp["flujo_neto_cc"].shift(-h)
        else:
            pq_banco = df_parquet[
                (df_parquet["banco"] == banco) & (df_parquet["h"] == h)
            ][["fecha_t", "target"]].rename(columns={"fecha_t": "fecha", "target": f"flujo_h{h}"})
            grp = grp.merge(pq_banco, on="fecha", how="left")

        for feat_col in FEATURES_CC:
            if feat_col not in grp.columns:
                continue
            sub = grp[[feat_col, f"flujo_h{h}"]].dropna()
            if len(sub) < 30:
                continue
            r = sub[feat_col].corr(sub[f"flujo_h{h}"])
            resultados_corr.append({
                "banco": banco, "feature": feat_col,
                "h": h, "correlacion": round(r, 3),
            })

df_corr = pd.DataFrame(resultados_corr)

if not df_corr.empty:
    pivot = df_corr.pivot_table(
        index=["banco", "feature"], columns="h", values="correlacion"
    ).rename(columns={h: f"h={h}" for h in H_EVALUAR})
    print("\n    Correlaciones por horizonte (h=1,2 omitidos — datos confirmados disponibles):")
    print(pivot.to_string())

    # ── Tabla separada solo para SISTEMA ────────────────────────────────────
    df_corr_sis = df_corr[df_corr["banco"] == "SISTEMA"]
    if not df_corr_sis.empty:
        print("\n    ── SISTEMA (suma 63 entidades) ──────────────────────────────")
        pivot_sis = df_corr_sis.pivot_table(
            index="feature", columns="h", values="correlacion"
        ).rename(columns={h: f"h={h}" for h in H_EVALUAR})
        print(pivot_sis.to_string())
    else:
        print("\n    SISTEMA no encontrado en parquet — verifica que 'banco'=='SISTEMA' exista.")

    # Gráfico: evolución de correlación por horizonte
    bancos_unicos = df_corr["banco"].unique()
    fig, axes = plt.subplots(1, len(bancos_unicos),
                             figsize=(5 * len(bancos_unicos), 5), sharey=True)
    if len(bancos_unicos) == 1:
        axes = [axes]
    for ax, banco_n in zip(axes, bancos_unicos):
        sub_b = df_corr[df_corr["banco"] == banco_n]
        for feat_col, sub_f in sub_b.groupby("feature"):
            sub_f = sub_f.sort_values("h")
            ax.plot(sub_f["h"], sub_f["correlacion"], marker="o", lw=1.5, label=feat_col)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_title(f"{banco_n}", fontweight="bold")
        ax.set_xlabel("Horizonte h (días hábiles)")
        ax.set_ylabel("Correlación con flujo neto")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    plt.suptitle(
        "Correlación features CC → flujo neto por horizonte\n"
        "(h≤2 omitidos — datos confirmados disponibles para esos horizontes)",
        fontweight="bold"
    )
    plt.tight_layout()
    ruta_fig3 = DIR_OUT / "03_correlacion_por_horizonte.png"
    plt.savefig(ruta_fig3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Guardado: {ruta_fig3.name}")

###############################################################################
# 5. EVOLUCIÓN TEMPORAL DE CC Y FLUJO NETO
###############################################################################

print("\n[4b] Graficando evolución de acumulados rolling por banco...")

BANCOS_PLOT = [b for b in ["BBVA", "SCOTIABANK", "INTERBANK", "CITIBANK"]
               if b in df_feat["banco"].unique()]
ROLLS = ["flujo_acum_roll_5d", "flujo_acum_roll_10d", "flujo_acum_roll_22d"]
COLORES_ROLL = {"flujo_acum_roll_5d": "steelblue",
                "flujo_acum_roll_10d": "darkorange",
                "flujo_acum_roll_22d": "seagreen"}

fig, axes = plt.subplots(len(BANCOS_PLOT), 1,
                         figsize=(15, 4 * len(BANCOS_PLOT)), sharex=True)
if len(BANCOS_PLOT) == 1:
    axes = [axes]

for ax, banco_n in zip(axes, BANCOS_PLOT):
    df_b = df_feat[df_feat["banco"] == banco_n].sort_values("fecha")
    for roll_col in ROLLS:
        if roll_col not in df_b.columns:
            continue
        n = roll_col.split("_")[-1]   # "5d", "10d", "22d"
        ax.plot(df_b["fecha"], df_b[roll_col] / 1e6,
                color=COLORES_ROLL[roll_col], lw=1.0, alpha=0.85, label=f"Roll {n}")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_title(f"{banco_n} — CC(t) − CC(t−N): variación neta rolling",
                 fontweight="bold", fontsize=9)
    ax.set_ylabel("MM USD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

plt.suptitle("Acumulado rolling de CC sin depender del inicio de mes\n"
             "Positivo → banco acumuló → probable retiro próximo",
             fontweight="bold")
plt.tight_layout()
ruta_fig4 = DIR_OUT / "04_acumulado_rolling_por_banco.png"
plt.savefig(ruta_fig4, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Guardado: {ruta_fig4.name}")

###############################################################################
# 5b. EVOLUCIÓN TEMPORAL DE CC_SISTEMA
###############################################################################

print("\n[4c] Graficando CC del SISTEMA (suma 63 entidades)...")

df_sis_plot = df_feat[df_feat["banco"] == "SISTEMA"].sort_values("fecha")

if not df_sis_plot.empty:
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    ax = axes[0]
    ax.plot(df_sis_plot["fecha"], df_sis_plot["cc"] / 1e6,
            color="steelblue", lw=1.2)
    ax.set_ylabel("Saldo CC (MM USD)")
    ax.set_title("CC del SISTEMA — suma de las 63 entidades", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    colores_sis = ["seagreen" if v >= 0 else "crimson"
                   for v in df_sis_plot["flujo_neto_cc"].fillna(0)]
    ax.bar(df_sis_plot["fecha"], df_sis_plot["flujo_neto_cc"].fillna(0) / 1e6,
           color=colores_sis, alpha=0.7, width=1)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("ΔCC_SISTEMA (MM USD)")
    ax.set_title("Flujo neto SISTEMA = ΔCC (target del modelo para banco=SISTEMA)",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[2]
    for roll_col, color in [("flujo_acum_roll_5d", "steelblue"),
                             ("flujo_acum_roll_10d", "darkorange"),
                             ("flujo_acum_roll_22d", "seagreen")]:
        if roll_col in df_sis_plot.columns:
            n = roll_col.split("_")[-1]
            ax.plot(df_sis_plot["fecha"], df_sis_plot[roll_col] / 1e6,
                    color=color, lw=1.0, alpha=0.85, label=f"Roll {n}")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_ylabel("CC(t)−CC(t−N) (MM USD)")
    ax.set_title("Acumulado rolling SISTEMA: CC(t)−CC(t−N)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    ruta_fig5 = DIR_OUT / "05_cc_sistema_evolucion.png"
    plt.savefig(ruta_fig5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Guardado: {ruta_fig5.name}")

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
    max_corr = df_corr["correlacion"].abs().max()
    umbral   = 0.10

    print(f"\n  Correlación máxima encontrada: {max_corr:.3f}  "
          f"(en h={df_corr.loc[df_corr['correlacion'].abs().idxmax(), 'h']})")
    if max_corr >= umbral:
        print(f"  → RECOMENDACIÓN: SÍ agregar a step001.")
        print(f"    Features candidatos (|corr| > {umbral:.0%} en algún horizonte):")
        top = df_corr.groupby(["banco","feature"])["correlacion"].apply(
            lambda x: x.abs().max()
        ).reset_index(name="max_abs_corr")
        top = top[top["max_abs_corr"] >= umbral].sort_values("max_abs_corr", ascending=False)
        for _, r in top.iterrows():
            print(f"      · {r['banco']:<12} {r['feature']:<25} max|corr|={r['max_abs_corr']:.3f}")
    else:
        print(f"  → RECOMENDACIÓN: EVALUAR. Correlación lineal baja (<{umbral:.0%}).")
        print(f"    XGBoost puede capturar relaciones no lineales — revisar los plots.")
else:
    print("\n  No se pudo calcular correlaciones. Revisa los plots generados.")

print(f"\n  Plots generados en: {DIR_OUT}")
print(f"  · 01_perfil_intramonth_cc.png      — hipótesis del encaje ({BANCO_FOCO})")
print(f"  · 02_evolucion_temporal_cc.png     — serie histórica + flujo + acumulado ({BANCO_FOCO})")
print(f"  · 03_correlacion_por_horizonte.png — correlación CC → flujo neto por banco")
print(f"  · 04_acumulado_rolling_por_banco.png — rolling 5d/10d/22d por banco")
print(f"  · 05_cc_sistema_evolucion.png      — CC agregada SISTEMA (63 entidades)")
print(f"  · features_cc_derivados.xlsx       — tabla completa incluye SISTEMA")
print()
