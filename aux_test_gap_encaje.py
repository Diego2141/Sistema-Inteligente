"""
¿bbva_encaje.xlsx tiene una fila por cada día CALENDARIO (incluidos fines de
semana) o solo por día HÁBIL? Decide entre las dos hipótesis planteadas para
capacidad_retiro_th, sin tocar ni modificar el archivo.

Hipótesis A — fila por día calendario (fin de semana con el saldo de BCR
    reportado, típicamente igual al del viernes por arrastre contable):
    capacidad_retiro_th ya está bien construida.
Hipótesis B — fila solo por día hábil, y la fila del lunes agrupa la
    acumulación de viernes+sábado+domingo en un solo valor:
    encaje_diario_lag1 hay que normalizarlo por el tamaño real del hueco
    antes de proyectarlo — ver la conversación para el fix exacto.

Qué mide
--------
1. Distribución de gaps entre fechas consecutivas (df["fecha"].diff().dt.days).
   Casi todo en 1 y algunos en 3/4 (findes de semana/feriados largos) → B.
   Presencia de gaps en 0 con fechas de sábado/domingo real → A.
2. Si hay fechas de sábado o domingo en el archivo, directamente.
3. Si el archivo SÍ tiene esas fechas: compara la magnitud de "encaje" en
   filas de sábado/domingo contra un día hábil típico. Si son prácticamente
   iguales (arrastre del viernes) confirma A con más fuerza; si son ~0 o
   discontinuas, podría ser un caso mixto a revisar aparte.

Uso: ejecutar directamente. Solo lee el Excel, no escribe nada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RUTA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\bbva_encaje.xlsx")


def main() -> None:
    if not RUTA.exists():
        raise FileNotFoundError(f"No se encuentra {RUTA}. Ajuste RUTA si el archivo está en otro lugar.")

    df = pd.read_excel(RUTA, header=0)
    df = df.iloc[:, :9].copy()
    df.columns = ["fecha", "entidad", "codigo", "overnight", "cta_cte",
                  "caja", "tose", "exigible", "retiro_neto"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["encaje"] = df["cta_cte"] + df["caja"]

    print(f"Archivo : {RUTA.name}")
    print(f"Filas   : {len(df):,}")
    print(f"Período : {df['fecha'].min().date()} → {df['fecha'].max().date()}")

    # ── 1) Distribución de gaps entre fechas consecutivas ──────────────────
    gaps = df["fecha"].diff().dt.days.dropna()
    print("\n" + "=" * 70)
    print("1 — Distribución de gaps entre filas consecutivas (días calendario)")
    print("=" * 70)
    vc = gaps.value_counts().sort_index()
    for g, n in vc.items():
        marca = ""
        if g == 1:
            marca = "  <- día siguiente inmediato"
        elif g in (2, 3):
            marca = "  <- podría ser fin de semana saltado"
        elif g >= 4:
            marca = "  <- feriado largo / hueco de datos"
        print(f"  gap={int(g):>3} días : {n:>6,} filas ({n/len(gaps):.1%}){marca}")

    # ── 2) ¿Hay fechas de sábado o domingo en el archivo? ───────────────────
    print("\n" + "=" * 70)
    print("2 — ¿Aparecen sábados o domingos como fecha propia?")
    print("=" * 70)
    dow = df["fecha"].dt.dayofweek  # 5=sábado, 6=domingo
    n_sab = int((dow == 5).sum())
    n_dom = int((dow == 6).sum())
    print(f"  Filas en sábado : {n_sab:,}")
    print(f"  Filas en domingo: {n_dom:,}")

    # ── 3) Si existen, comparar su magnitud contra un día hábil típico ──────
    if n_sab + n_dom > 0:
        print("\n" + "=" * 70)
        print("3 — Magnitud de 'encaje' en sábado/domingo vs. día hábil típico")
        print("=" * 70)
        habil = df.loc[dow < 5, "encaje"]
        finde = df.loc[dow >= 5, "encaje"]
        print(f"  encaje hábil  — mediana: {habil.median():,.0f}  "
              f"P10-P90: [{habil.quantile(.1):,.0f}, {habil.quantile(.9):,.0f}]")
        print(f"  encaje finde  — mediana: {finde.median():,.0f}  "
              f"P10-P90: [{finde.quantile(.1):,.0f}, {finde.quantile(.9):,.0f}]")

        # ¿El sábado/domingo repite el valor del viernes anterior? (arrastre)
        df_idx = df.set_index("fecha")["encaje"]
        finde_fechas = df.loc[dow >= 5, "fecha"]
        n_match, n_tot = 0, 0
        for f in finde_fechas:
            viernes = f - pd.Timedelta(days=(f.dayofweek - 4))
            if viernes in df_idx.index:
                n_tot += 1
                if np.isclose(df_idx.get(f, np.nan), df_idx.get(viernes, np.nan), rtol=1e-6):
                    n_match += 1
        if n_tot > 0:
            print(f"\n  Coincide con el viernes anterior (arrastre exacto): "
                  f"{n_match}/{n_tot} ({n_match/n_tot:.0%})")

    # ── Veredicto ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Veredicto")
    print("=" * 70)
    frac_gap1 = (gaps == 1).mean()
    if n_sab + n_dom > 0 and frac_gap1 > 0.9:
        print("  → HIPÓTESIS A: el archivo tiene fila por día calendario.")
        print("    capacidad_retiro_th ya está bien construida, no hace falta tocar nada.")
    elif n_sab + n_dom == 0:
        print("  → HIPÓTESIS B: no hay filas de sábado/domingo — el archivo salta")
        print("    fines de semana. Revisar si 'exigible'/'encaje' en la fila del lunes")
        print("    ya vienen agregados (comparar su magnitud contra un día hábil normal:")
        print("    si el lunes es sistemáticamente ~3x un martes/miércoles típico, están")
        print("    agregados y hace falta normalizar encaje_diario_lag1 por el tamaño")
        print("    del hueco antes de proyectarlo en capacidad_retiro_th.")
    else:
        print("  → Resultado mixto — revisar la sección 1 y 3 en detalle antes de decidir.")


if __name__ == "__main__":
    main()
