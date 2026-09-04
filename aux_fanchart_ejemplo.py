# -*- coding: utf-8 -*-
"""
aux_fanchart_ejemplo.py
Fan chart para una fecha calendario fija (6 de julio) a través de todos los
años disponibles en la muestra.

Para cada año, traza la ruta realizada de D-R desde el 6 de julio (o el
día hábil más cercano si cae en fin de semana/feriado) hasta h=90 días
hábiles hacia adelante, superpuesta sobre las bandas históricas del banco.

Permite ver si existe un patrón estacional recurrente para esa fecha.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_ejemplo"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

# Bancos a graficar
BANCOS      = ["CREDITO", "BBVA", "SCOTIABANK"]
DIA_FIJO    = (7, 6)    # (mes, día) → 6 de julio

# ─────────────────────────────────────────────────────────────────
# 1. Leer columnas clave
# ─────────────────────────────────────────────────────────────────
print("Cargando columnas clave...")
df_key = pd.read_parquet(RUTA_MATRIZ, columns=["fecha_t", "banco", "h", "target"])
df_key["fecha_t"] = pd.to_datetime(df_key["fecha_t"])

# ─────────────────────────────────────────────────────────────────
# 2. Para cada año, encontrar el día hábil más cercano al 6-jul
# ─────────────────────────────────────────────────────────────────
def nearest_bday(year, mes, dia, fechas_disponibles):
    """
    Retorna la fecha hábil más cercana a (year-mes-dia) dentro de
    fechas_disponibles. Busca hasta ±5 días alrededor del objetivo.
    """
    objetivo = pd.Timestamp(year, mes, dia)
    candidatos = [objetivo + pd.Timedelta(days=d) for d in range(-5, 6)]
    for c in sorted(candidatos, key=lambda x: abs((x - objetivo).days)):
        if c in fechas_disponibles:
            return c
    return None


mes_fijo, dia_fijo = DIA_FIJO

# ─────────────────────────────────────────────────────────────────
# 3. Generar un fan chart por banco
# ─────────────────────────────────────────────────────────────────
for banco in BANCOS:
    if banco not in df_key["banco"].unique():
        print(f"  {banco} no encontrado en la matriz — omitiendo")
        continue

    sub = df_key[(df_key["banco"] == banco) & df_key["target"].notna()].copy()
    fechas_disponibles = set(sub["fecha_t"].unique())
    años_disponibles   = sorted(sub["fecha_t"].dt.year.unique())

    # Buscar el día hábil más cercano al 6-jul en cada año
    fechas_julio = {}
    for año in años_disponibles:
        f = nearest_bday(año, mes_fijo, dia_fijo, fechas_disponibles)
        if f is not None:
            # Incluir si tiene al menos 20% de los horizontes disponibles con target
            # (umbral relativo: funciona aunque la muestra no llegue al futuro lejano)
            h_max_global = sub["h"].max()
            n_h = sub[(sub["fecha_t"] == f) & sub["target"].notna()]["h"].count()
            if n_h >= max(10, int(h_max_global * 0.20)):
                fechas_julio[año] = f

    if not fechas_julio:
        print(f"  {banco}: no hay fechas julio disponibles con cobertura suficiente")
        continue

    print(f"\n{banco}: {len(fechas_julio)} años con 6-jul hábil")
    for año, f in fechas_julio.items():
        print(f"  {año} → {f.date()}")

    # Distribución histórica por h (bandas del fondo)
    tgt_h = sub.groupby("h")["target"]
    hs    = sorted(sub["h"].unique())
    q05   = tgt_h.quantile(0.05) / 1e6
    q25   = tgt_h.quantile(0.25) / 1e6
    q50   = tgt_h.median()       / 1e6
    q75   = tgt_h.quantile(0.75) / 1e6
    q95   = tgt_h.quantile(0.95) / 1e6

    # Colores por año
    n_años  = len(fechas_julio)
    cmap    = cm.get_cmap("tab20" if n_años <= 20 else "turbo", n_años)
    colores = {año: cmap(i) for i, año in enumerate(sorted(fechas_julio))}

    # ── Gráfico ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7))

    # Bandas históricas
    ax.fill_between(hs, q05[hs], q95[hs],
                    alpha=0.10, color="steelblue", label="P5–P95 hist.")
    ax.fill_between(hs, q25[hs], q75[hs],
                    alpha=0.25, color="steelblue", label="P25–P75 hist.")
    ax.plot(hs, q50[hs], color="steelblue", lw=1.5, ls="--",
            alpha=0.7, label="Mediana hist.")

    # Ruta realizada para cada año
    for año, fecha in sorted(fechas_julio.items()):
        ruta = (
            sub[sub["fecha_t"] == fecha]
            .sort_values("h")
            .set_index("h")["target"] / 1e6
        )
        ax.plot(ruta.index, ruta.values,
                color=colores[año], lw=1.6, alpha=0.85,
                label=f"{año} ({fecha.strftime('%d %b')})", zorder=4)

    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)

    # Anotación: aprox. dónde cae Fiestas Patrias (28 jul ≈ h=16 días hábiles)
    ax.axvspan(14, 18, alpha=0.06, color="red", label="~Fiestas Patrias (28 jul)")

    ax.set_xlabel("Horizonte h (días hábiles desde el 6 de julio)", fontsize=10)
    ax.set_ylabel("Flujo neto D−R (MM USD)", fontsize=10)
    ax.set_title(
        f"Fan Chart — {banco}  |  Fecha de inicio: 6 de julio (cada año)\n"
        f"Ruta realizada h = {min(hs)}…{max(hs)} días hábiles",
        fontweight="bold", fontsize=11
    )
    ax.legend(fontsize=8, ncol=3, loc="upper right",
              framealpha=0.9, borderpad=0.6)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(min(hs) - 1, max(hs) + 1)
    plt.tight_layout()

    nombre = f"fanchart_{banco}_julio06_todos_anios.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Guardado: {DIR_OUTPUT / nombre}")

print(f"\nFan charts guardados en: {DIR_OUTPUT}")
