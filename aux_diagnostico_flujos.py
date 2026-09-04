# -*- coding: utf-8 -*-
"""
aux_diagnostico_flujos.py
=========================
Reconcilia Transacciones_BancaLocal.xlsx (crudo) contra el pivot que produce
load_manual_data() en step001, paso por paso, para responder por qué los flujos
de la matriz no cuadran con la data en bruto y por qué aparecen más fechas.

Mide CUATRO fugas posibles, todas silenciosas en el código actual:

  A. Montos no parseables       to_numeric(errors="coerce") los vuelve NaN y
                                groupby.sum() los saltea. Nadie los cuenta.
  B. Fechas fuera de Lun-Vie    reindex(bdate_range) DESCARTA las filas cuya
                                fecha no cae en día hábil Lun-Vie. Si hay
                                transacciones en fin de semana, su flujo
                                desaparece del pivot sin aviso.
  C. Feriados rellenados con 0  bdate_range NO excluye feriados peruanos, así
                                que ~291 fechas entran al índice con R=D=0.
                                Son indistinguibles de un día hábil sin
                                transacciones, y arrastran hacia abajo todas
                                las medias y volatilidades móviles.
  D. Cobertura por banco        un banco que no operó en parte del período
                                recibe 0 en vez de NaN, o sea historia
                                fabricada en sus features.

Uso (desde Spyder, sin argumentos):
    runfile('aux_diagnostico_flujos.py')

Sin acceso a H: corre un autotest sintético que verifica que el diagnóstico
detecta cada fuga cuando existe.
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd

RUTA_EXCEL = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
              r"\1. Data\Raw\Transacciones_BancaLocal.xlsx")

COL_FECHA = "Fecha Valor"
COL_BANCO = "Broker"
COL_MONTO = "Delivery Principal Usd"

_ALIAS = {"CONTINEN": "BBVA"}   # mismo que PARAMS["alias_bancos"] de step001


def _sep(t):
    print("\n" + "=" * 74); print(f"  {t}"); print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
def diagnosticar(df_raw, feriados=None):
    """
    Recibe el Excel crudo ya leído y devuelve un dict con las métricas de cada
    fuga. Replica exactamente los pasos de load_manual_data() para que la
    comparación sea contra lo que el pipeline realmente hace.
    """
    rep = {}
    n0 = len(df_raw)
    rep["filas_crudas"] = n0

    # ── A. Montos no parseables ─────────────────────────────────────────────
    monto = pd.to_numeric(df_raw[COL_MONTO], errors="coerce")
    malos = monto.isna() & df_raw[COL_MONTO].notna()
    rep["monto_no_parseable"] = int(malos.sum())
    rep["monto_vacio"] = int(df_raw[COL_MONTO].isna().sum())
    rep["ejemplos_no_parseables"] = (
        df_raw.loc[malos, COL_MONTO].astype(str).unique()[:5].tolist())

    df = df_raw.copy()
    df["_fecha"] = pd.to_datetime(df[COL_FECHA], errors="coerce")
    rep["fecha_no_parseable"] = int(df["_fecha"].isna().sum())
    df["_monto"] = monto
    df["_R"] = df["_monto"].clip(upper=0).abs()
    df["_D"] = df["_monto"].clip(lower=0)

    rep["R_total_crudo"] = float(df["_R"].sum())
    rep["D_total_crudo"] = float(df["_D"].sum())

    # ── B. Fechas que el reindex descartaría ────────────────────────────────
    val = df.dropna(subset=["_fecha"])
    fechas = pd.DatetimeIndex(val["_fecha"].unique()).sort_values()
    idx_bd = pd.bdate_range(start=fechas.min(), end=fechas.max())   # Lun-Vie
    fuera = fechas.difference(idx_bd)
    rep["fechas_totales"] = len(fechas)
    rep["fechas_fuera_de_lun_vie"] = len(fuera)
    rep["fechas_fuera_lista"] = [str(d.date()) for d in fuera[:10]]
    perdido = val[val["_fecha"].isin(fuera)]
    rep["R_perdido_por_reindex"] = float(perdido["_R"].sum())
    rep["D_perdido_por_reindex"] = float(perdido["_D"].sum())
    rep["filas_perdidas_por_reindex"] = len(perdido)
    rep["dias_semana_fuera"] = (
        perdido["_fecha"].dt.day_name().value_counts().to_dict() if len(perdido) else {})

    # ── C. Fechas que el reindex INVENTA ────────────────────────────────────
    inventadas = idx_bd.difference(fechas)
    rep["fechas_inventadas"] = len(inventadas)
    if feriados is not None and len(inventadas):
        fer = pd.DatetimeIndex(feriados)
        rep["inventadas_que_son_feriado"] = int(inventadas.isin(fer).sum())
    else:
        rep["inventadas_que_son_feriado"] = None

    # ── D. Cobertura por banco ──────────────────────────────────────────────
    val = val.copy()
    val["_banco"] = val[COL_BANCO].astype(str).apply(
        lambda b: _ALIAS.get(b.upper(), b))
    cob = []
    for b, g in val.groupby("_banco"):
        f0, f1 = g["_fecha"].min(), g["_fecha"].max()
        habiles_en_rango = len(pd.bdate_range(f0, f1))
        cob.append({
            "banco": b,
            "primera": f0.date(), "ultima": f1.date(),
            "dias_con_dato": g["_fecha"].nunique(),
            "dias_habiles_del_rango": habiles_en_rango,
            "ceros_fuera_de_su_rango": len(idx_bd) - habiles_en_rango,
        })
    rep["cobertura"] = pd.DataFrame(cob).sort_values("ceros_fuera_de_su_rango",
                                                     ascending=False)

    # ── Reconciliación final: pivot como lo arma step001 ────────────────────
    agg = (val.groupby(["_banco", "_fecha"])[["_R", "_D"]].sum().reset_index())
    wide = agg.pivot_table(index="_fecha", columns="_banco",
                           values=["_R", "_D"], aggfunc="sum")
    wide.columns = [f"{c[1]}_{c[0][1:]}" for c in wide.columns]
    wide = wide.sort_index()
    idx_completo = pd.bdate_range(start=wide.index.min(), end=wide.index.max())
    wide_re = wide.reindex(idx_completo, fill_value=0.0).fillna(0.0)

    rc = [c for c in wide_re.columns if c.endswith("_R")]
    dc = [c for c in wide_re.columns if c.endswith("_D")]
    rep["R_total_pivot"] = float(wide_re[rc].to_numpy().sum())
    rep["D_total_pivot"] = float(wide_re[dc].to_numpy().sum())
    rep["fechas_pivot"] = len(wide_re)
    return rep


def imprimir(rep):
    _sep("A · MONTOS QUE NO SE PUDIERON PARSEAR")
    print(f"  filas crudas                : {rep['filas_crudas']:,}")
    print(f"  monto vacío (NaN de origen) : {rep['monto_vacio']:,}")
    print(f"  monto NO parseable          : {rep['monto_no_parseable']:,}"
          + ("   <-- SE PIERDEN SIN AVISO" if rep["monto_no_parseable"] else ""))
    if rep["ejemplos_no_parseables"]:
        print(f"  ejemplos                    : {rep['ejemplos_no_parseables']}")
    print(f"  fecha NO parseable          : {rep['fecha_no_parseable']:,}")

    _sep("B · FECHAS DESCARTADAS POR reindex(bdate_range)")
    print(f"  fechas únicas en el crudo   : {rep['fechas_totales']:,}")
    print(f"  fuera de Lun-Vie            : {rep['fechas_fuera_de_lun_vie']:,}")
    if rep["fechas_fuera_de_lun_vie"]:
        print(f"  primeras                    : {rep['fechas_fuera_lista']}")
        print(f"  por día de semana           : {rep['dias_semana_fuera']}")
        print(f"  filas perdidas              : {rep['filas_perdidas_por_reindex']:,}")
        print(f"  R perdido                   : {rep['R_perdido_por_reindex']:>18,.2f}")
        print(f"  D perdido                   : {rep['D_perdido_por_reindex']:>18,.2f}")
        print("  *** Esas transacciones NO llegan a la matriz. ***")
    else:
        print("  OK: ninguna transacción cae fuera de Lun-Vie.")

    _sep("C · FECHAS QUE EL reindex AGREGA CON R=D=0")
    print(f"  fechas en el pivot          : {rep['fechas_pivot']:,}")
    print(f"  agregadas (sin dato crudo)  : {rep['fechas_inventadas']:,}")
    if rep["inventadas_que_son_feriado"] is not None:
        print(f"  de ellas, feriados PE/USA   : {rep['inventadas_que_son_feriado']:,}")
    print("  Un feriado con R=D=0 es indistinguible de un día hábil sin")
    print("  transacciones, y arrastra hacia abajo ma_flujo_* y sigma_flujo_*.")

    _sep("D · COBERTURA POR BANCO (ceros fabricados fuera de su período)")
    cob = rep["cobertura"]
    print(cob.head(12).to_string(index=False))
    parciales = cob[cob["ceros_fuera_de_su_rango"] > 250]
    if len(parciales):
        print(f"\n  {len(parciales)} banco(s) con más de un año de ceros fabricados")
        print("  fuera de su período real. Sus medias y volatilidades móviles")
        print("  están calculadas sobre historia que no existió.")

    _sep("RECONCILIACIÓN CRUDO vs PIVOT")
    for k in ("R", "D"):
        c, p = rep[f"{k}_total_crudo"], rep[f"{k}_total_pivot"]
        d = p - c
        estado = "OK" if abs(d) < 0.01 else "*** NO CUADRA ***"
        print(f"  {k}: crudo {c:>18,.2f} | pivot {p:>18,.2f} | "
              f"dif {d:>+15,.2f}  {estado}")


# ─────────────────────────────────────────────────────────────────────────────
def autotest():
    """
    Verifica que el diagnóstico DETECTA cada fuga, construyendo un caso que la
    contiene a propósito. Un diagnóstico que nunca puede reportar un problema
    no sirve para nada.
    """
    print("Sin acceso al Excel: corriendo autotest sintético.\n")
    filas = []
    # dos días hábiles normales
    for f in ("2024-01-08", "2024-01-09"):
        filas += [{COL_FECHA: f, COL_BANCO: "BBVA",    COL_MONTO: -100.0},
                  {COL_FECHA: f, COL_BANCO: "CREDITO", COL_MONTO:  250.0}]
    # A: monto no parseable
    filas.append({COL_FECHA: "2024-01-10", COL_BANCO: "BBVA", COL_MONTO: "1.234,56"})
    # B: transacción en SABADO -> el reindex la descarta
    filas.append({COL_FECHA: "2024-01-13", COL_BANCO: "BBVA", COL_MONTO: -999.0})
    # D: banco que solo aparece al final del período
    filas.append({COL_FECHA: "2024-02-15", COL_BANCO: "NUEVO", COL_MONTO: -50.0})
    df = pd.DataFrame(filas)

    rep = diagnosticar(df)
    imprimir(rep)

    _sep("AUTOTEST: ¿detectó cada fuga?")
    ok = True
    for nombre, cond, det in [
        ("A monto no parseable", rep["monto_no_parseable"] == 1,
         f"detectados {rep['monto_no_parseable']}"),
        ("B fecha fuera de Lun-Vie", rep["fechas_fuera_de_lun_vie"] == 1,
         f"detectadas {rep['fechas_fuera_de_lun_vie']}"),
        ("B flujo perdido > 0", rep["R_perdido_por_reindex"] == 999.0,
         f"R perdido = {rep['R_perdido_por_reindex']}"),
        ("C fechas inventadas", rep["fechas_inventadas"] > 0,
         f"{rep['fechas_inventadas']} fechas"),
        ("D banco de cobertura parcial", len(rep["cobertura"]) == 3,
         f"{len(rep['cobertura'])} bancos"),
        ("reconciliación NO cuadra",
         abs(rep["R_total_pivot"] - rep["R_total_crudo"]) > 0.01,
         "el sábado perdido rompe el total, como debe"),
    ]:
        print(f"  [{'OK ' if cond else 'FALLA'}] {nombre} — {det}")
        ok &= cond
    print("\n" + ("Todas las fugas se detectan correctamente."
                  if ok else "*** El diagnóstico NO detecta alguna fuga ***"))
    return 0 if ok else 1


def main():
    p = pathlib.Path(RUTA_EXCEL)
    if not p.exists():
        sys.exit(autotest())

    print(f"Leyendo {p.name} ...")
    df_raw = pd.read_excel(p)
    print(f"  {len(df_raw):,} filas × {df_raw.shape[1]} columnas")
    faltan = [c for c in (COL_FECHA, COL_BANCO, COL_MONTO) if c not in df_raw.columns]
    if faltan:
        print(f"\nFaltan columnas esperadas: {faltan}")
        print(f"Disponibles: {list(df_raw.columns)}")
        sys.exit(1)

    feriados = None
    try:
        os.environ.setdefault("BCRP_PROXY", "http://diagnostico-sin-red")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_s1", pathlib.Path(__file__).with_name("step001_build_feature_matrix_v2.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _, feriados = m.construir_calendario_habil(m.PARAMS)[:2]
    except Exception as e:
        print(f"  (no se pudo cargar el calendario de feriados: {e})")

    imprimir(diagnosticar(df_raw, feriados))


if __name__ == "__main__":
    main()
