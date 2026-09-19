# -*- coding: utf-8 -*-
"""
aux_diagnostico_overlay.py
==========================
Genera un Excel con toda la informacion intermedia del overlay sobreencaje T+2,
fecha a fecha, para verificar que la logica aplicada tiene sentido economico.

COMO LEER EL EXCEL
------------------
Sheet "Resumen":
  Una fila por cada fecha_t del periodo TEST. Muestra:
  - Que cierre trimestral se usa (y si se salto el primero por T+2)
  - Cuantos dias de la ventana son conocidos vs inciertos
  - El netting: retiro_conocido descuenta del cap (peor_total)
  - El peor_restante que es el numerador real del factor f

Sheet "Detalle_ventana":
  Una fila por cada (fecha_t, dia_calendario) dentro de la ventana de 7dh.
  Muestra el flujo real de cada dia y su clasificacion:
    - pasado:          dia ya ocurrido (antes de fecha_t)
    - conocido_T+N:    dia futuro conocido con antelacion (h=1,2)
    - incierto:        dia genuinamente incierto (h>=3 hasta h_cierre)

Sheet "Metodologia":
  Descripcion del concepto y formula.

INTUICION
---------
peor_total actua como un TOPE (cap) sobre el retiro acumulado en 7dh.
Si el peor historico fue 4 B USD, los bancos no pueden retirar mas de eso
(en escenario extremo). A medida que los dias transcurren, los retiros
reales van "consumiendo" ese tope. Lo que queda es el margen residual.

Ejemplo: peor_total = 4B. A 3 dias del cierre, ya se retiraron 2.8B
en los 4 dias anteriores (conocidos). Solo quedan 1.2B de margen.
El factor f = 1.2B / |Q05 proyectado para esos 3 dias|.

La regla T+2 agrega: los 2 proximos dias habiles son conocidos de antemano
(los bancos notifican con ese plazo). Si el cierre esta a <= 2 dias,
todo es determinista -> se salta al siguiente trimestre.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas (deben coincidir con step005_walk_forward_cv_3.py) ──────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
RUTA_AJUSTE  = BASE_SISTEMA / "2. Output" / "analisis_cc" / "saldos_retiros_bancos.xlsx"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "analisis_cc"

BANCO      = "SISTEMA"
CORTE_TEST = pd.Timestamp("2024-01-26")   # inicio del periodo TEST fold 1

# Parametros overlay (deben coincidir con step005_walk_forward_cv_3.py)
OVERLAY_VENTANA_DH              = 7
OVERLAY_CONOCIMIENTO_ANTICIPADO = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_flow(df_h1: pd.Series, d: pd.Timestamp) -> float:
    """Flujo neto D-R del dia d, buscando target h=1 del dia anterior."""
    d_prev = d - pd.offsets.BDay(1)
    if d_prev in df_h1.index:
        return float(df_h1.loc[d_prev])
    return np.nan


def _cierres_trimestre(bh_list: list, df_bh: pd.DataFrame) -> list[tuple]:
    """Retorna lista de (fecha_cierre, h_cierre) ordenada por h."""
    cierres_fecha = (
        df_bh[df_bh["mes"].isin([3, 6, 9, 12])]
        .groupby(["anio", "trim"])["fecha"].max()
        .sort_values().tolist()
    )
    result = []
    for cand in cierres_fecha:
        try:
            h = bh_list.index(cand) + 1
        except ValueError:
            diffs = [abs((d - cand).days) for d in bh_list]
            h = diffs.index(min(diffs)) + 1
        result.append((cand, h))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N   = OVERLAY_CONOCIMIENTO_ANTICIPADO
    VDH = OVERLAY_VENTANA_DH

    print("Cargando datos...")
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", BANCO)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])

    # Serie rapida: target del dia siguiente (h=1) indexada por fecha_t
    df_h1 = df[df["h"] == 1].set_index("fecha_t")["target"]

    df_aj = pd.read_excel(
        RUTA_AJUSTE, sheet_name="Ajuste_diario", index_col=0, parse_dates=True
    )
    df_aj.index = pd.DatetimeIndex(df_aj.index)

    fechas_test = sorted(
        df[df["fecha_t"] >= CORTE_TEST]["fecha_t"].unique()
    )
    print(f"Fechas TEST: {len(fechas_test)}")

    rows_res  = []   # Sheet Resumen
    rows_det  = []   # Sheet Detalle_ventana

    for fecha_t in fechas_test:
        fecha_t = pd.Timestamp(fecha_t)

        # peor_total vigente en fecha_t
        disps = df_aj.index[df_aj.index <= fecha_t]
        if len(disps) == 0:
            continue
        peor_total = float(df_aj.loc[disps[-1], "peor_total"])
        if peor_total <= 0:
            continue

        # Horizonte de dias habiles desde fecha_t
        bh = pd.bdate_range(
            start=fecha_t + pd.offsets.BDay(1),
            periods=N + VDH + 75 + 5,
        )
        bh_list = list(bh)
        df_bh = pd.DataFrame({
            "fecha": bh, "mes": bh.month, "anio": bh.year, "trim": bh.quarter
        })
        cierres = _cierres_trimestre(bh_list, df_bh)
        if not cierres:
            continue

        primer_cierre_fecha, primer_cierre_h = cierres[0]
        primer_saltado = primer_cierre_h <= N

        # Seleccionar primer cierre con horizonte incierto > N
        cierre_usado_fecha = cierre_usado_h = None
        for (cand, h) in cierres:
            if h > N:
                cierre_usado_fecha, cierre_usado_h = cand, h
                break
        if cierre_usado_fecha is None:
            continue

        h_inicio = max(1, cierre_usado_h - VDH + 1)
        fecha_inicio_ventana = cierre_usado_fecha - pd.offsets.BDay(VDH - 1)
        fecha_conocida_hasta = min(
            fecha_t + pd.offsets.BDay(N),
            cierre_usado_fecha,
        )
        h_inicio_incierto = max(h_inicio, N + 1)
        n_dias_inciertos   = max(0, cierre_usado_h - h_inicio_incierto + 1)

        # Todos los dias de la ventana de 7dh
        all_window = pd.bdate_range(start=fecha_inicio_ventana, end=cierre_usado_fecha)

        # Dias conocidos (pasados + T+N)
        if fecha_inicio_ventana <= fecha_conocida_hasta:
            known_days = set(pd.bdate_range(start=fecha_inicio_ventana, end=fecha_conocida_hasta))
        else:
            known_days = set()

        # Flujos y netting
        retiro_conocido = 0.0
        for d in all_window:
            flow = _get_flow(df_h1, d)

            # Clasificacion del dia
            if d <= fecha_t:
                tipo = "pasado"
                h_d  = -(len(pd.bdate_range(start=d, end=fecha_t)) - 1)
            elif d in known_days:
                tipo = "conocido_T+N"
                h_d  = bh_list.index(d) + 1 if d in bh_list else None
            else:
                tipo = "incierto"
                h_d  = bh_list.index(d) + 1 if d in bh_list else None

            contribuye = (d in known_days) and (not np.isnan(flow)) and (flow < 0)
            retiro_parcial = abs(flow) / 1e6 if contribuye else 0.0
            if contribuye:
                retiro_conocido += abs(flow)

            rows_det.append({
                "fecha_t"           : fecha_t.date(),
                "cierre_trimestral" : cierre_usado_fecha.date(),
                "dia_calendario"    : d.date(),
                "h_desde_t"         : h_d,
                "tipo"              : tipo,
                "flujo_neto_MM"     : round(flow / 1e6, 2) if not np.isnan(flow) else None,
                "es_retiro_neto"    : bool(flow < 0) if not np.isnan(flow) else None,
                "contribuye_netting": contribuye,
                "retiro_parcial_MM" : round(retiro_parcial, 2),
            })

        peor_restante = max(0.0, peor_total - retiro_conocido)
        cap_consumido = peor_restante < 1e-6

        if primer_saltado:
            razon = (
                f"h_cierre={primer_cierre_h} <= T+N={N}: "
                f"cierre del {primer_cierre_fecha.date()} ya conocido"
            )
        else:
            razon = ""

        rows_res.append({
            "fecha_t"                : fecha_t.date(),
            # --- Primer cierre ---
            "primer_cierre_fecha"    : primer_cierre_fecha.date(),
            "primer_cierre_h"        : primer_cierre_h,
            "primer_cierre_saltado"  : primer_saltado,
            "razon_salto"            : razon,
            # --- Cierre efectivo ---
            "cierre_usado_fecha"     : cierre_usado_fecha.date(),
            "cierre_usado_h"         : cierre_usado_h,
            # --- Ventana ---
            "ventana_inicio_fecha"   : fecha_inicio_ventana.date(),
            "conocida_hasta_fecha"   : fecha_conocida_hasta.date(),
            "n_dias_ventana_total"   : len(all_window),
            "n_dias_conocidos"       : len(known_days & set(all_window)),
            "n_dias_inciertos"       : n_dias_inciertos,
            # --- Netting ---
            "retiro_conocido_MM"     : round(retiro_conocido / 1e6, 2),
            "peor_total_MM"          : round(peor_total / 1e6, 2),
            "peor_restante_MM"       : round(peor_restante / 1e6, 2),
            "cap_consumido"          : cap_consumido,
            # --- Ventana incierta (denominador del factor) ---
            "h_inicio_incierto"      : h_inicio_incierto,
            "h_cierre_incierto"      : cierre_usado_h,
            # Q05_acum viene del modelo (step005); no disponible aqui
            "Q05_acum_MM_MODELO"     : None,
            "factor_f_MODELO"        : None,
            "nota"                   : (
                "cap consumido: sin ajuste" if cap_consumido
                else f"factor = peor_restante / |Q05_acum[h={h_inicio_incierto}..{cierre_usado_h}]|"
            ),
        })

    # ── Hoja Metodologia ──────────────────────────────────────────────────────
    rows_met = [
        {"concepto": "peor_total",
         "descripcion": "Peor retiro acumulado historico en 7dh hábiles antes del cierre trimestral (de step007). Actúa como tope máximo."},
        {"concepto": "retiro_conocido",
         "descripcion": "Suma de retiros netos ya conocidos en la ventana de 7dh: dias pasados (realizados) + proximos T+N dias habiles (notificados por los bancos)."},
        {"concepto": "peor_restante",
         "descripcion": "max(0, peor_total - retiro_conocido). Lo que todavia puede retirarse segun el historial extremo."},
        {"concepto": "Skip h_cierre <= N",
         "descripcion": f"Si el cierre esta a {N} o menos dias habiles, todos los flujos son conocidos -> no hay incertidumbre -> se salta al siguiente cierre trimestral."},
        {"concepto": "Q05_acum",
         "descripcion": f"Suma del cuantil Q05 del modelo sobre la ventana incierta [h_inicio_incierto .. h_cierre]. Denominador del factor."},
        {"concepto": "factor_f",
         "descripcion": "peor_restante / |Q05_acum|. Solo se aplica si f > 1. Se multiplica uniformemente a todos los cuantiles y a toda la ventana de proyeccion."},
        {"concepto": "Intuicion",
         "descripcion": "El cap se 'consume' a medida que los dias transcurren. Si ya se retiro casi todo el maximo historico, el factor f converge a 1 (sin ajuste adicional). Si el cierre es lejano y no hubo retiros aun, f refleja la brecha entre el peor escenario y la proyeccion del modelo."},
        {"concepto": "T+2 advance",
         "descripcion": "Los bancos notifican con 2 dias habiles de antelacion. Esos 2 dias se incluyen en 'retiro_conocido' (no como proyeccion sino como hecho), reduciendo la incertidumbre residual y estabilizando el factor."},
    ]

    # ── Exportar ──────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows_res)
    df_det = pd.DataFrame(rows_det)
    df_met = pd.DataFrame(rows_met)

    ruta_out = DIR_OUTPUT / "diagnostico_overlay_T2.xlsx"
    print(f"\nEscribiendo Excel: {ruta_out} ...")
    with pd.ExcelWriter(ruta_out, engine="openpyxl") as writer:
        df_met.to_excel(writer, sheet_name="Metodologia",     index=False)
        df_res.to_excel(writer, sheet_name="Resumen",         index=False)
        df_det.to_excel(writer, sheet_name="Detalle_ventana", index=False)

    # ── Resumen consola ───────────────────────────────────────────────────────
    total    = len(df_res)
    saltados = df_res["primer_cierre_saltado"].sum()
    con_net  = (df_res["retiro_conocido_MM"] > 0).sum()
    cap_cons = df_res["cap_consumido"].sum()

    print(f"\n{'='*55}")
    print(f"  Fechas TEST analizadas    : {total:>6,}")
    print(f"  Cierres saltados (T+N)    : {saltados:>6,}  "
          f"({100*saltados/total:.1f}%)")
    print(f"  Con netting > 0           : {con_net:>6,}  "
          f"({100*con_net/total:.1f}%)")
    print(f"  Cap consumido (sin ajuste): {cap_cons:>6,}  "
          f"({100*cap_cons/total:.1f}%)")
    print(f"{'='*55}")
    print(f"\nExcel guardado en:\n  {ruta_out}")


if __name__ == "__main__":
    main()
