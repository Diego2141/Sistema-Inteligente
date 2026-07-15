# -*- coding: utf-8 -*-
"""
aux_dias_no_laborables.py
Diagnóstico de días no laborables adicionales presentes en los datos bancarios.

Detecta fechas donde SISTEMA_R == 0 Y SISTEMA_D == 0 simultáneamente
y que NO pertenecen al calendario estándar PE+USA. Estos son "puentes"
declarados por decreto supremo (DS) que no figuran en ningún calendario
programático pero que el sistema bancario no operó.

Uso: python aux_dias_no_laborables.py
Salida: tabla en consola + Excel en 2. Output/
"""

import os
import getpass
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
BASE       = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_BANC  = BASE / "1. Data" / "Raw" / "Transacciones_BancaLocal.xlsx"
RUTA_OUT   = BASE / "2. Output" / "dias_no_laborables.xlsx"
AÑOS_CAL   = list(range(2009, 2042))
# ─────────────────────────────────────────────────────────────────────────────


# ── Calendario PE+USA (idéntico a step001) ───────────────────────────────────
def build_peru_holidays(años):
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, Holiday, GoodFriday,
        USFederalHolidayCalendar, Easter,
    )
    from pandas.tseries.offsets import Day as _Day

    f0, f1 = f"{min(años)}-01-01", f"{max(años)}-12-31"

    _easter_dates = [
        pd.Timestamp(y, m, d) for y, m, d in [
            (2009,4,12),(2010,4,4),(2011,4,24),(2012,4,8),(2013,3,31),
            (2014,4,20),(2015,4,5),(2016,3,27),(2017,4,16),(2018,4,1),
            (2019,4,21),(2020,4,12),(2021,4,4),(2022,4,17),(2023,4,9),
            (2024,3,31),(2025,4,20),(2026,4,5),(2027,3,28),(2028,4,16),
            (2029,4,1),(2030,4,21),(2031,4,13),(2032,3,28),(2033,4,17),
            (2034,4,9),(2035,3,25),(2036,4,13),(2037,4,5),(2038,4,25),
            (2039,4,10),(2040,4,1),(2041,4,21),
        ]
    ]

    class PEUSCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday("Año Nuevo",        month=1,  day=1),
            Holiday("Jueves Santo",     month=1,  day=1,
                    offset=[Easter(), _Day(-3)]),
            GoodFriday,
            Holiday("Día del Trabajo",  month=5,  day=1),
            Holiday("San Pedro y Pablo",month=6,  day=29),
            Holiday("Fiestas Patrias 1",month=7,  day=28),
            Holiday("Fiestas Patrias 2",month=7,  day=29),
            Holiday("Santa Rosa",       month=8,  day=30),
            Holiday("Combate Angamos", month=10,  day=8),
            Holiday("Todos los Santos", month=11,  day=1),
            Holiday("Inmaculada Concepción",month=12,day=8),
            Holiday("Navidad",          month=12, day=25),
            # US
            Holiday("New Year US",      month=1,  day=1),
            Holiday("MLK Day",          month=1,  day=1,
                    offset=pd.DateOffset(weekday=0, n=3)),    # 3er lunes enero
            Holiday("Presidents Day",   month=2,  day=1,
                    offset=pd.DateOffset(weekday=0, n=3)),
            Holiday("Memorial Day",     month=5,  day=25,
                    offset=pd.DateOffset(weekday=0)),
            Holiday("Juneteenth",       month=6,  day=19),
            Holiday("Independence Day", month=7,  day=4),
            Holiday("Labor Day US",     month=9,  day=1,
                    offset=pd.DateOffset(weekday=0)),
            Holiday("Columbus Day",     month=10, day=1,
                    offset=pd.DateOffset(weekday=0, n=2)),
            Holiday("Veterans Day",     month=11, day=11),
            Holiday("Thanksgiving",     month=11, day=22,
                    offset=pd.DateOffset(weekday=3)),
            Holiday("Christmas US",     month=12, day=25),
        ]

    cal  = PEUSCalendar()
    hols = cal.holidays(start=f0, end=f1)

    # Agregar Jueves/Viernes Santo hardcodeados
    for e in _easter_dates:
        hols = hols.append(pd.DatetimeIndex([e - pd.Timedelta(days=3), e - pd.Timedelta(days=2)]))

    return hols.normalize().unique().sort_values()


# ── Cargar bancarios ─────────────────────────────────────────────────────────
def cargar_bancarios(ruta):
    if not os.environ.get("BCRP_PROXY"):
        print("=" * 55)
        _u = input("  Usuario de red (ej: 2579): ").strip()
        _p = getpass.getpass("  Contraseña: ")
        os.environ["BCRP_PROXY"] = f"http://{_u}:{_p}@bcrproxy:8080"

    print(f"\nCargando {ruta.name}...")
    df = pd.read_excel(ruta, sheet_name=0)

    # Detectar columna de fecha
    fecha_col = next(
        (c for c in df.columns if "fecha" in str(c).lower() or "date" in str(c).lower()),
        df.columns[0],
    )
    df[fecha_col] = pd.to_datetime(df[fecha_col]).dt.normalize()

    # Pivot: bancos en columnas, R y D separados
    banco_col = next(c for c in df.columns if "banco" in str(c).lower())
    r_col = next(c for c in df.columns if str(c).upper() in ("R", "RETIRO", "RETIROS"))
    d_col = next(c for c in df.columns if str(c).upper() in ("D", "DEPOSITO", "DEPOSITOS"))

    df_r = df.pivot_table(index=fecha_col, columns=banco_col, values=r_col, aggfunc="sum").fillna(0)
    df_d = df.pivot_table(index=fecha_col, columns=banco_col, values=d_col, aggfunc="sum").fillna(0)
    df_r.columns = [f"{c}_R" for c in df_r.columns]
    df_d.columns = [f"{c}_D" for c in df_d.columns]

    df_wide = pd.concat([df_r, df_d], axis=1).fillna(0)
    idx = pd.bdate_range(start=df_wide.index.min(), end=df_wide.index.max())
    df_wide = df_wide.reindex(idx, fill_value=0.0)
    df_wide.index.name = "fecha"

    # SISTEMA = suma
    r_cols = [c for c in df_wide.columns if c.endswith("_R")]
    d_cols = [c for c in df_wide.columns if c.endswith("_D")]
    df_wide["SISTEMA_R"] = df_wide[r_cols].sum(axis=1)
    df_wide["SISTEMA_D"] = df_wide[d_cols].sum(axis=1)

    print(f"  {df_wide.shape[0]:,} fechas, {len(r_cols)} bancos")
    return df_wide


# ── Detección ────────────────────────────────────────────────────────────────
def detectar_dias_no_laborables(df_banc, peru_holidays):
    zero_mask  = (df_banc["SISTEMA_R"] == 0) & (df_banc["SISTEMA_D"] == 0)
    zero_dates = df_banc.index[zero_mask].normalize()
    hols_norm  = peru_holidays.normalize()
    extra      = zero_dates[~zero_dates.isin(hols_norm)]
    return extra.sort_values()


# ── Enriquecer con contexto ──────────────────────────────────────────────────
def enriquecer(extra, df_banc, peru_holidays):
    registros = []
    for fecha in extra:
        dia_sem = fecha.strftime("%A")
        anterior = df_banc.index[df_banc.index < fecha]
        siguiente = df_banc.index[df_banc.index > fecha]
        sis_r_ant = df_banc.loc[anterior[-1], "SISTEMA_R"] if len(anterior) else np.nan
        sis_d_ant = df_banc.loc[anterior[-1], "SISTEMA_D"] if len(anterior) else np.nan
        sis_r_sig = df_banc.loc[siguiente[0], "SISTEMA_R"] if len(siguiente) else np.nan
        sis_d_sig = df_banc.loc[siguiente[0], "SISTEMA_D"] if len(siguiente) else np.nan

        # ¿Hay feriado estándar adyacente?
        hols = peru_holidays.normalize()
        dia_ant = (fecha - pd.Timedelta(days=1)).normalize()
        dia_sig = (fecha + pd.Timedelta(days=1)).normalize()
        feriado_ant = dia_ant in hols or (fecha - pd.Timedelta(days=3)).normalize() in hols
        feriado_sig = dia_sig in hols or (fecha + pd.Timedelta(days=1)).normalize() in hols

        if feriado_sig:
            tipo = "Puente (antes de feriado)"
        elif feriado_ant:
            tipo = "Puente (después de feriado)"
        elif fecha.weekday() in (3, 4):  # Jue/Vie
            tipo = "Puente (largo fin de semana)"
        else:
            tipo = "Decreto especial"

        registros.append({
            "fecha":              fecha.date(),
            "dia_semana":         dia_sem,
            "tipo_estimado":      tipo,
            "SISTEMA_R_dia_ant":  sis_r_ant,
            "SISTEMA_D_dia_ant":  sis_d_ant,
            "SISTEMA_R_dia_sig":  sis_r_sig,
            "SISTEMA_D_dia_sig":  sis_d_sig,
        })
    return pd.DataFrame(registros)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    peru_holidays = build_peru_holidays(AÑOS_CAL)
    df_banc       = cargar_bancarios(RUTA_BANC)
    extra         = detectar_dias_no_laborables(df_banc, peru_holidays)

    print(f"\n{'='*65}")
    print(f"  DÍAS NO LABORABLES ADICIONALES DETECTADOS: {len(extra)}")
    print(f"  (SISTEMA_R=0 Y SISTEMA_D=0, fuera del calendario PE+USA)")
    print(f"{'='*65}")

    if len(extra) == 0:
        print("  Ninguno detectado.")
    else:
        df_out = enriquecer(extra, df_banc, peru_holidays)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:,.0f}".format)
        print(df_out.to_string(index=False))

        print(f"\n  → {len(extra)} fechas serán anuladas en bancarios y excluidas de fecha_t")
        print(f"  → Para añadir manualmente (residuos no cero): PARAMS['dias_no_habiles_adicionales']")

        # Guardar Excel
        df_out.to_excel(RUTA_OUT, index=False)
        print(f"\n  Excel guardado: {RUTA_OUT}")

    # Lista para copiar a PARAMS (por si se necesita override manual)
    print(f"\n  Lista para PARAMS['dias_no_habiles_adicionales']:")
    print("  [")
    for f in extra:
        print(f'      "{f.date()}",')
    print("  ]")
