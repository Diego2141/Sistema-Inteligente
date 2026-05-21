# -*- coding: utf-8 -*-
"""
aux_fanchart_ejemplo.py
Fan chart para una fecha de predicción aleatoria: muestra la ruta realizada
de flujos futuros (D-R) superpuesta sobre la distribución histórica del banco.

Ilustra por qué la distribución marginal es plana (promedia estacionalidad)
pero para una fecha concreta sí se observan patrones estacionales.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_ejemplo"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

# Bancos a graficar (los de mayor volumen y variabilidad según boxplot)
BANCOS  = ["CREDITO", "BBVA", "SCOTIABANK"]
SEED    = 42
rng     = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────
# 1. Leer columnas clave (liviano)
# ─────────────────────────────────────────────────────────────────
print("Cargando columnas clave...")
df_key = pd.read_parquet(RUTA_MATRIZ, columns=["fecha_t", "banco", "h", "target"])
df_key["fecha_t"] = pd.to_datetime(df_key["fecha_t"])

# ─────────────────────────────────────────────────────────────────
# 2. Leer features estacionales para anotar el gráfico
# ─────────────────────────────────────────────────────────────────
cols_cal = ["fecha_t", "banco", "h",
            "is_ultimo_bday_trim", "is_cierre_encaje",
            "is_pre_feriado", "mes", "pos_en_mes"]
df_cal = pd.read_parquet(RUTA_MATRIZ, columns=cols_cal)
df_cal["fecha_t"] = pd.to_datetime(df_cal["fecha_t"])

# ─────────────────────────────────────────────────────────────────
# 3. Elegir fecha_t aleatoria con cobertura completa de h y target
# ─────────────────────────────────────────────────────────────────
def elegir_fecha(df, banco):
    """Fecha aleatoria donde todos los horizontes tienen target realizado."""
    sub = df[(df["banco"] == banco) & df["target"].notna()]
    h_max = sub["h"].max()
    fechas_ok = sub.groupby("fecha_t")["h"].count()
    fechas_ok = fechas_ok[fechas_ok == h_max - sub["h"].min() + 1].index
    # Evitar extremos del historial (sin warm-up ni fin de serie)
    fechas_ok = fechas_ok[(fechas_ok.year >= 2012) & (fechas_ok.year <= 2024)]
    return pd.Timestamp(rng.choice(fechas_ok))


# ─────────────────────────────────────────────────────────────────
# 4. Generar un fan chart por banco
# ─────────────────────────────────────────────────────────────────
for banco in BANCOS:
    if banco not in df_key["banco"].unique():
        print(f"  {banco} no encontrado en la matriz — omitiendo")
        continue

    fecha = elegir_fecha(df_key, banco)
    print(f"\n{banco}: fecha elegida = {fecha.date()}")

    # Distribución histórica del banco por horizonte h
    sub_hist = df_key[(df_key["banco"] == banco) & df_key["target"].notna()]
    tgt_h    = sub_hist.groupby("h")["target"]
    hs       = sorted(sub_hist["h"].unique())

    q05 = tgt_h.quantile(0.05) / 1e6
    q25 = tgt_h.quantile(0.25) / 1e6
    q50 = tgt_h.median()       / 1e6
    q75 = tgt_h.quantile(0.75) / 1e6
    q95 = tgt_h.quantile(0.95) / 1e6

    # Ruta realizada desde la fecha elegida
    realizado = (
        sub_hist[sub_hist["fecha_t"] == fecha]
        .sort_values("h")
        .set_index("h")["target"] / 1e6
    )

    # Features estacionales en esa ruta (para anotaciones)
    cal_fecha = (
        df_cal[(df_cal["banco"] == banco) & (df_cal["fecha_t"] == fecha)]
        .sort_values("h")
        .set_index("h")
    )

    # ── Gráfico ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.fill_between(hs, q05[hs], q95[hs],
                    alpha=0.12, color="steelblue", label="P5–P95 hist.")
    ax.fill_between(hs, q25[hs], q75[hs],
                    alpha=0.30, color="steelblue", label="P25–P75 hist.")
    ax.plot(hs, q50[hs], color="steelblue", lw=1.5, ls="--",
            alpha=0.8, label="Mediana hist.")

    ax.plot(realizado.index, realizado.values,
            color="darkorange", lw=2.2, zorder=5,
            label=f"Realizado desde {fecha.strftime('%d %b %Y')}")
    ax.scatter(realizado.index, realizado.values,
               color="darkorange", s=10, zorder=6)

    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)

    # Anotar eventos estacionales en la ruta realizada
    if not cal_fecha.empty:
        for h_val, row in cal_fecha.iterrows():
            if h_val not in realizado.index:
                continue
            y_val = realizado.get(h_val, np.nan)
            if pd.isna(y_val):
                continue
            if row.get("is_ultimo_bday_trim", 0) == 1:
                ax.axvline(h_val, color="purple", lw=0.8, ls=":", alpha=0.7)
                ax.text(h_val + 0.3, ax.get_ylim()[1] * 0.92,
                        "Fin Trim", fontsize=6.5, color="purple", rotation=90, va="top")
            elif row.get("is_cierre_encaje", 0) == 1:
                ax.axvline(h_val, color="green", lw=0.8, ls=":", alpha=0.6)
                ax.text(h_val + 0.3, ax.get_ylim()[1] * 0.92,
                        "Cierre\nmes", fontsize=6, color="green", rotation=90, va="top")

    ax.set_xlabel("Horizonte h (días hábiles desde la fecha de predicción)",
                  fontsize=10)
    ax.set_ylabel("Flujo neto D−R (MM USD)", fontsize=10)
    ax.set_title(
        f"Fan Chart Histórico — {banco}\n"
        f"Fecha de predicción: {fecha.strftime('%d %B %Y')} | "
        f"h = {min(hs)} … {max(hs)} días hábiles",
        fontweight="bold", fontsize=11
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(min(hs) - 1, max(hs) + 1)
    plt.tight_layout()

    nombre = f"fanchart_{banco}_{fecha.strftime('%Y%m%d')}.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Guardado: {DIR_OUTPUT / nombre}")

print(f"\nFan charts guardados en: {DIR_OUTPUT}")
