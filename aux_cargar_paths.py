# -*- coding: utf-8 -*-
"""
aux_cargar_paths.py — carga los dos simulacion_paths_*.parquet en memoria para
mirarlos desde el Variable Explorer de Spyder.

    runfile("aux_cargar_paths.py")

Deja seis objetos en el namespace global:

    df_conjunto        tabla larga de CONJUNTO_BBVA   (317.016 x 5)
    df_sistema         tabla larga de SISTEMA         ( 30.114 x 5)
    fan_conjunto       un ORIGEN: ventana x tau       (74 x 10)
    fan_sistema        un ORIGEN: ventana x tau       ( 7 x 10)
    serie_conjunto     una VENTANA: fecha_t x tau     (476 x 10)
    serie_sistema      una VENTANA: fecha_t x tau     (478 x 10)

POR QUE LAS CUATRO VISTAS Y NO SOLO LAS DOS TABLAS
El formato largo es el correcto para guardar y para que otro lo consuma, pero
317.016 filas de (fecha_t, ventana, tau) no se "ven": hay que scrollear 74 filas
para completar un solo origen. Las dos preguntas que uno le hace a este archivo
tienen forma de tabla ancha:

  - "como es el abanico en una fecha" -> pivot ventana x tau  (fan_*)
  - "como evoluciona la banda a un plazo" -> pivot fecha_t x tau  (serie_*)

Las dos salen de pivotar la misma tabla larga, sin recalcular nada.

Las rutas y resolver_ruta se IMPORTAN de aux_verificar_estructura_paths.py en vez
de repetirse: si alguien mueve los archivos, se edita un solo lugar y los dos
scripts lo siguen. Es el mismo criterio con el que etiqueta_corrida() esta
definida igual en step006 y en generar_video_fancharts.
"""

from pathlib import Path

import pandas as pd

try:
    from aux_verificar_estructura_paths import (
        ARCHIVOS, COLUMNAS_CONTRATO, resolver_ruta, describir,
    )
except ImportError as e:
    raise ImportError(
        "Falta aux_verificar_estructura_paths.py en la misma carpeta — de ahi "
        "salen las rutas y resolver_ruta. Copialo junto a este archivo en vez de "
        "duplicar las rutas aca."
    ) from e

###############################################################################
# Configuración
###############################################################################

# Etiqueta corta por archivo, en el mismo orden que ARCHIVOS. Solo para nombrar
# las variables y los prints.
ETIQUETAS = ["conjunto", "sistema"]

# El origen y la ventana de las vistas anchas. None = se elige automaticamente:
#   fecha  -> la ULTIMA disponible (el abanico mas reciente, que es el que se
#             mira para decidir)
#   ventana-> 22 si existe (un mes habil), si no la mediana de la grilla
FECHA_VISTA   = None
VENTANA_VISTA = None

# Exportar a Excel. Por defecto solo las vistas ANCHAS, que son chicas
# (74x10 y 476x10) y son lo que uno abre.
#
# Las tablas largas quedan afuera a proposito: 317.016 filas por openpyxl son
# minutos de escritura y un archivo que Excel abre con esfuerzo, para mostrar lo
# mismo que el Variable Explorer ya tiene cargado. Si igual hacen falta, poner
# EXPORTAR_LARGAS = True y preferir CSV.
EXPORTAR_VISTAS  = True
EXPORTAR_LARGAS  = False
DIR_EXPORT       = Path(".")          # cwd; en Spyder, el wdir del runfile


###############################################################################
# Carga y vistas
###############################################################################

def cargar(ruta_cadena: str) -> pd.DataFrame:
    """Lee un simulacion_paths_*.parquet y normaliza fecha_t a datetime."""
    ruta = resolver_ruta(ruta_cadena)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")
    df = pd.read_parquet(ruta)
    faltan = [c for c in COLUMNAS_CONTRATO if c not in df.columns]
    if faltan:
        raise ValueError(f"{ruta.name}: faltan columnas del contrato {faltan}. "
                         f"Tiene {list(df.columns)}")
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df.attrs["ruta"] = str(ruta)
    df.attrs["nombre"] = ruta.name
    return df


def fan_de_origen(df: pd.DataFrame, fecha=None) -> pd.DataFrame:
    """
    El abanico de UN origen: filas = ventana, columnas = los 9 tau.

    Se lee como el fan chart: bajando por las filas se avanza en el plazo, y de
    izquierda a derecha se abre la banda. La columna y_realizado va al final
    para poder comparar cada fila con lo que efectivamente paso.
    """
    fecha = df["fecha_t"].max() if fecha is None else pd.Timestamp(fecha)
    sub = df[df["fecha_t"] == fecha]
    if sub.empty:
        raise ValueError(f"No hay filas con fecha_t={fecha:%Y-%m-%d}. "
                         f"Rango disponible: {df['fecha_t'].min():%Y-%m-%d} .. "
                         f"{df['fecha_t'].max():%Y-%m-%d}")
    ancho = sub.pivot(index="ventana", columns="tau", values="percentil_acum")
    ancho.columns = [f"q{int(round(t * 100)):02d}" for t in ancho.columns]
    ancho["y_realizado"] = (sub.drop_duplicates("ventana")
                            .set_index("ventana")["y_realizado_acum"])
    ancho.attrs["fecha_t"] = fecha
    return ancho.sort_index()


def serie_de_ventana(df: pd.DataFrame, ventana=None) -> pd.DataFrame:
    """
    La banda a UN plazo a lo largo del tiempo: filas = fecha_t, columnas = tau.

    Es la vista del backtest: y_realizado contra q05 fila por fila dice cuando
    hubo excedencia.
    """
    vs = sorted(df["ventana"].unique().tolist())
    if ventana is None:
        ventana = 22 if 22 in vs else vs[len(vs) // 2]
    ventana = int(ventana)
    if ventana not in vs:
        raise ValueError(f"ventana={ventana} no esta. Disponibles: {vs}")
    sub = df[df["ventana"] == ventana]
    ancho = sub.pivot(index="fecha_t", columns="tau", values="percentil_acum")
    ancho.columns = [f"q{int(round(t * 100)):02d}" for t in ancho.columns]
    ancho["y_realizado"] = (sub.drop_duplicates("fecha_t")
                            .set_index("fecha_t")["y_realizado_acum"])
    ancho.attrs["ventana"] = ventana
    return ancho.sort_index()


def resumen(etiqueta: str, df: pd.DataFrame) -> None:
    """Lo que uno quiere saber antes de mirar: forma, grillas, nulos, cabecera."""
    d = describir(df)
    print(f"\n{'=' * 78}\n{etiqueta.upper()}  —  {df.attrs.get('nombre', '?')}")
    print(f"{'=' * 78}")
    print(f"  forma      : {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"  memoria    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  columnas   : {list(df.columns)}")
    print(f"  dtypes     : {d['dtypes']}")
    print(f"  taus ({len(d['taus'])})   : {d['taus']}")
    vs = sorted(df["ventana"].unique().tolist())
    print(f"  ventanas ({len(vs)}): {vs if len(vs) <= 12 else f'{vs[:6]} ... {vs[-3:]}'}")
    f0, f1, nf = d["fechas"]
    print(f"  origenes   : {nf} fechas, {f0:%Y-%m-%d} .. {f1:%Y-%m-%d}")

    # La grilla completa se verifica, no se asume: si el producto no da, hay
    # combinaciones faltantes y cualquier pivot va a salir con NaN.
    esperadas = nf * len(vs) * len(d["taus"])
    estado = "completa" if esperadas == len(df) else f"INCOMPLETA (faltan {esperadas - len(df):,})"
    print(f"  grilla     : {nf} x {len(vs)} x {len(d['taus'])} = {esperadas:,} -> {estado}")

    n_nan = int(df["y_realizado_acum"].isna().sum())
    if n_nan:
        # Esperable al final de la muestra: un origen reciente todavia no tiene
        # 75 dias habiles de realizado por delante.
        ult = df.loc[df["y_realizado_acum"].isna(), "fecha_t"]
        print(f"  y_realizado: {n_nan:,} NaN ({n_nan / len(df):.1%}), "
              f"desde {ult.min():%Y-%m-%d}")
    else:
        print("  y_realizado: sin NaN")
    print("\n  primeras filas:")
    print(df.head(6).to_string(index=False))


def exportar(pares: dict, dir_salida: Path) -> None:
    """
    Escribe las vistas anchas a Excel, una hoja por vista.

    Envuelto en try/except a proposito: la exportacion es el ULTIMO paso, y si
    revienta —falta openpyxl, el .xlsx esta abierto en Excel y Windows lo tiene
    tomado, no hay permiso de escritura en el wdir— se lleva puesta la corrida
    entera y los DataFrames ya cargados no llegan al Variable Explorer. El
    objetivo del script es verlos; el Excel es una comodidad.
    """
    ruta = dir_salida / "vistas_simulacion_paths.xlsx"
    try:
        dir_salida.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            for nombre, obj in pares.items():
                # Excel corta los nombres de hoja en 31 caracteres sin avisar, y
                # dos hojas que colisionan tras el corte hacen fallar la escritura.
                obj.to_excel(xl, sheet_name=nombre[:31])
        print(f"\n  -> {ruta}  ({len(pares)} hojas: {', '.join(pares)})")
    except ImportError:
        print("\n  [AVISO] falta openpyxl, no se exporto el Excel "
              "(pip install openpyxl). Los DataFrames quedan cargados igual.")
    except OSError as e:
        print(f"\n  [AVISO] no se pudo escribir {ruta} ({type(e).__name__}: {e}). "
              f"Si lo tenes abierto en Excel, cerralo. Los DataFrames quedan "
              f"cargados igual.")


###############################################################################
# Main
###############################################################################

def main():
    dfs, fans, series = {}, {}, {}

    for etq, ruta in zip(ETIQUETAS, ARCHIVOS):
        df = cargar(ruta)
        resumen(etq, df)
        dfs[etq] = df
        fans[etq] = fan_de_origen(df, FECHA_VISTA)
        series[etq] = serie_de_ventana(df, VENTANA_VISTA)

        f = fans[etq].attrs["fecha_t"]
        v = series[etq].attrs["ventana"]
        print(f"\n  fan_{etq}   : abanico del origen {f:%Y-%m-%d}  "
              f"{fans[etq].shape[0]} ventanas x {fans[etq].shape[1]} columnas")
        print(fans[etq].head(5).to_string())
        print(f"\n  serie_{etq} : banda a ventana={v}  "
              f"{series[etq].shape[0]} fechas x {series[etq].shape[1]} columnas")
        print(series[etq].head(5).to_string())

    if EXPORTAR_VISTAS:
        pares = {f"fan_{k}": v for k, v in fans.items()}
        pares.update({f"serie_{k}": v for k, v in series.items()})
        exportar(pares, DIR_EXPORT)

    if EXPORTAR_LARGAS:
        for etq, df in dfs.items():
            ruta = DIR_EXPORT / f"simulacion_paths_{etq}.csv"
            df.to_csv(ruta, index=False)
            print(f"  -> {ruta}  ({len(df):,} filas)")

    print(f"\n{'=' * 78}")
    print("En el Variable Explorer: " + ", ".join(
        [f"df_{k}" for k in dfs] + [f"fan_{k}" for k in fans]
        + [f"serie_{k}" for k in series]))
    print("=" * 78)
    return dfs, fans, series


if __name__ == "__main__":
    # Las asignaciones van al namespace GLOBAL a proposito: con runfile() de
    # Spyder eso es lo que hace que aparezcan en el Variable Explorer. Si la
    # carga viviera dentro de main() los DataFrames moririan al volver.
    _dfs, _fans, _series = main()

    df_conjunto    = _dfs["conjunto"]
    df_sistema     = _dfs["sistema"]
    fan_conjunto   = _fans["conjunto"]
    fan_sistema    = _fans["sistema"]
    serie_conjunto = _series["conjunto"]
    serie_sistema  = _series["sistema"]
