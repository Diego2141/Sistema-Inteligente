# -*- coding: utf-8 -*-
"""
build_feature_matrix.py
Construye la matriz de features para el sistema de predicción de liquidez
en moneda extranjera del banco central peruano.

Modelo: Gradient Boosting con quantile regression (LightGBM).
h (horizonte) es un feature más → UN SOLO modelo por banco.
Los bancos pequeños se agrupan en "Otros_bancos" antes de construir features.
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging con timestamps
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# PARTE 0 — Parámetros globales
###############################################################################
PARAMS = {
    # Fechas
    "fecha_inicio_historico": "2015-01-01",
    "fecha_fin_historico": "2025-12-31",

    # Modelo
    "h_min": 2,
    "h_max": 90,
    "quantiles": [0.01, 0.05, 0.50, 0.95, 0.99],

    # Modo demo (True mientras no lleguen los datos reales)
    "usar_datos_demo": True,

    # Agrupación de bancos
    "umbral_banco_pequeño_pct": 0.05,   # bancos con < 5% del volumen → Otros
    "bancos_otros": [],                  # lista fija (tiene prioridad si no está vacía)
    "nombre_otros": "Otros_bancos",

    # Rezagos bancarios
    "lags_cortos": [1, 2, 3],
    "lag_semana": 5,
    "lag_mes": 22,
    "ventanas_vol": [5, 22],

    # Calendario
    "años_calendario": list(range(2015, 2031)),

    # Rutas archivos manuales
    "ruta_datos_bancarios": r"RUTA\datos_bancarios.xlsx",
    "ruta_confirmados": r"RUTA\confirmados.xlsx",
    "ruta_intervencion": r"RUTA\intervencion.xlsx",
    "ruta_igv": r"RUTA\fechas_igv.xlsx",
    "ruta_elecciones": r"RUTA\fechas_elecciones.xlsx",
    "ruta_aux_xgboost": r"RUTA\Aux_XGBoost.py",
    "ruta_output": r"RUTA\matriz_features.xlsx",

    # APIs externas
    "fred_api_key": "TU_API_KEY",
    "proxy": "http://USUARIO:PASSWORD@bcrproxy:8080",

    # Códigos BCRP
    "bcrp_embi": "PD04638DD",
    "bcrp_tasa_ref": "PD04739DD",
    "bcrp_tc": "PD04648DD",

    # Tickers Yahoo Finance
    "ticker_vix": "^VIX",
    "ticker_tc": "PEN=X",
    "ticker_t10y": "^TNX",

    # Código FRED
    "fred_fedfunds": "FEDFUNDS",
}


###############################################################################
# PARTE 1 — Calendario hábil peruano
###############################################################################
def build_peru_calendar(años, ruta_igv, ruta_elecciones):
    """
    Retorna (peru_bday, peru_holidays, fechas_igv, fechas_elecciones).

    Si los archivos de IGV o elecciones no existen o están vacíos,
    retorna listas vacías sin lanzar error. Imprime advertencia.
    """
    logger.info("PARTE 1: Construyendo calendario hábil peruano...")

    try:
        import holidays
        from pandas.tseries.offsets import CustomBusinessDay

        peru_hols = holidays.Peru(years=años)
        peru_holidays = pd.to_datetime(list(peru_hols.keys()))
        peru_bday = CustomBusinessDay(holidays=peru_holidays)
        logger.info(f"  Feriados cargados: {len(peru_holidays)} días en {min(años)}-{max(años)}")
    except ImportError:
        logger.warning("  Librería 'holidays' no instalada. Usando calendario sin feriados.")
        from pandas.tseries.offsets import BDay
        peru_bday = BDay()
        peru_holidays = pd.DatetimeIndex([])

    # Cargar fechas IGV
    fechas_igv = []
    try:
        df_igv = pd.read_excel(ruta_igv)
        if not df_igv.empty:
            col = df_igv.columns[0]
            fechas_igv = pd.to_datetime(df_igv[col].dropna()).tolist()
            logger.info(f"  Fechas IGV cargadas: {len(fechas_igv)}")
        else:
            logger.warning(f"  Archivo IGV vacío: {ruta_igv}")
    except Exception:
        logger.warning(f"  No se pudo cargar fechas IGV desde: {ruta_igv}")

    # Cargar fechas elecciones
    fechas_elecciones = []
    try:
        df_elec = pd.read_excel(ruta_elecciones)
        if not df_elec.empty:
            col = df_elec.columns[0]
            fechas_elecciones = pd.to_datetime(df_elec[col].dropna()).tolist()
            logger.info(f"  Fechas elecciones cargadas: {len(fechas_elecciones)}")
        else:
            logger.warning(f"  Archivo elecciones vacío: {ruta_elecciones}")
    except Exception:
        logger.warning(f"  No se pudo cargar fechas elecciones desde: {ruta_elecciones}")

    logger.info("  Calendario construido correctamente.")
    return peru_bday, peru_holidays, fechas_igv, fechas_elecciones


def get_future_business_dates(t, h_max, peru_bday):
    """Retorna los h_max días hábiles siguientes desde t (excluye t)."""
    fechas = pd.date_range(start=t + peru_bday, periods=h_max, freq=peru_bday)
    return fechas


###############################################################################
# PARTE 2 — Carga de datos manuales
###############################################################################
def load_manual_data(params):
    """
    Carga datos manuales desde archivos Excel definidos en PARAMS.

    Si un archivo no está disponible, carga un DataFrame vacío y muestra advertencia.
    Retorna dict con claves: 'bancarios', 'confirmados', 'intervencion'.
    """
    logger.info("PARTE 2: Cargando datos manuales...")

    resultado = {
        "bancarios": pd.DataFrame(),
        "confirmados": pd.DataFrame(columns=["banco", "R_conf_t1", "R_conf_t2", "D_conf_t1"]),
        "intervencion": pd.Series(dtype=float),
    }

    # Datos bancarios
    try:
        df = pd.read_excel(params["ruta_datos_bancarios"])
        if df.empty:
            logger.warning(f"  Archivo bancarios vacío: {params['ruta_datos_bancarios']}")
        else:
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"])
                df = df.set_index("fecha").sort_index()
            resultado["bancarios"] = df
            logger.info(f"  Datos bancarios cargados: {df.shape}")
    except Exception as e:
        logger.warning(f"  No se pudo cargar datos bancarios: {params['ruta_datos_bancarios']} | {e}")

    # Confirmados
    try:
        df = pd.read_excel(params["ruta_confirmados"])
        if df.empty:
            logger.warning(f"  Archivo confirmados vacío: {params['ruta_confirmados']}")
        else:
            resultado["confirmados"] = df
            logger.info(f"  Datos confirmados cargados: {df.shape}")
    except Exception as e:
        logger.warning(f"  No se pudo cargar confirmados: {params['ruta_confirmados']} | {e}")

    # Intervención cambiaria
    try:
        df = pd.read_excel(params["ruta_intervencion"])
        if df.empty:
            logger.warning(f"  Archivo intervención vacío: {params['ruta_intervencion']}")
        else:
            fecha_col = df.columns[0]
            val_col = df.columns[1]
            serie = df.set_index(fecha_col)[val_col]
            serie.index = pd.to_datetime(serie.index)
            resultado["intervencion"] = serie.sort_index()
            logger.info(f"  Intervención cambiaria cargada: {len(serie)} observaciones")
    except Exception as e:
        logger.warning(f"  No se pudo cargar intervención: {params['ruta_intervencion']} | {e}")

    logger.info("  Carga de datos manuales completada.")
    return resultado


###############################################################################
# PARTE 3 — Agrupación de bancos
###############################################################################
def agrupar_bancos(df_bancarios, umbral_pct, bancos_otros, nombre_otros):
    """
    Agrupa bancos pequeños en una categoría única antes de modelar.

    Retorna (df_agrupado, lista_bancos, reporte).
    """
    logger.info("PARTE 3: Agrupando bancos pequeños...")

    if df_bancarios.empty:
        logger.warning("  DataFrame bancarios vacío. No hay bancos que agrupar.")
        return df_bancarios, [], {}

    # Detectar bancos disponibles (columnas tipo banco_X_R y banco_X_D)
    cols_R = [c for c in df_bancarios.columns if c.endswith("_R")]
    cols_D = [c for c in df_bancarios.columns if c.endswith("_D")]
    bancos_disponibles = [c[:-2] for c in cols_R if c[:-2] + "_D" in cols_D]

    if not bancos_disponibles:
        logger.warning("  No se detectaron columnas con formato {banco}_R / {banco}_D.")
        return df_bancarios, [], {}

    # Determinar bancos pequeños
    if bancos_otros:
        # Opción B: lista fija
        pequeños = [b for b in bancos_otros if b in bancos_disponibles]
        logger.info(f"  Usando lista fija de bancos pequeños: {pequeños}")
    else:
        # Opción A: umbral automático por volumen
        volumen = {}
        for b in bancos_disponibles:
            r = df_bancarios.get(f"{b}_R", pd.Series(dtype=float))
            d = df_bancarios.get(f"{b}_D", pd.Series(dtype=float))
            volumen[b] = (r.abs() + d.abs()).mean()

        total_vol = sum(volumen.values())
        participacion = {b: v / total_vol for b, v in volumen.items()} if total_vol > 0 else {b: 0 for b in bancos_disponibles}
        pequeños = [b for b, p in participacion.items() if p < umbral_pct]
        logger.info(f"  Bancos detectados automáticamente como pequeños (< {umbral_pct:.0%}): {pequeños}")

    grandes = [b for b in bancos_disponibles if b not in pequeños]

    df_agrupado = df_bancarios.copy()

    if pequeños:
        # Sumar R y D de bancos pequeños
        cols_R_pequeños = [f"{b}_R" for b in pequeños if f"{b}_R" in df_agrupado.columns]
        cols_D_pequeños = [f"{b}_D" for b in pequeños if f"{b}_D" in df_agrupado.columns]

        df_agrupado[f"{nombre_otros}_R"] = df_agrupado[cols_R_pequeños].sum(axis=1)
        df_agrupado[f"{nombre_otros}_D"] = df_agrupado[cols_D_pequeños].sum(axis=1)

        cols_a_eliminar = cols_R_pequeños + cols_D_pequeños
        df_agrupado = df_agrupado.drop(columns=cols_a_eliminar)

        lista_bancos = grandes + [nombre_otros]
    else:
        lista_bancos = grandes

    # Reporte
    reporte = {
        "bancos_individuales": grandes,
        "bancos_agrupados": pequeños,
        "nombre_grupo": nombre_otros,
    }

    if not df_bancarios.empty and not bancos_otros:
        reporte["participaciones"] = participacion if "participacion" in dir() else {}

    # Imprimir tabla resumen
    logger.info(f"\n{'='*55}")
    logger.info(f"  RESUMEN DE AGRUPACIÓN DE BANCOS")
    logger.info(f"{'='*55}")
    logger.info(f"  Bancos individuales ({len(grandes)}): {grandes}")
    logger.info(f"  Bancos agrupados en '{nombre_otros}' ({len(pequeños)}): {pequeños}")
    logger.info(f"  Total entidades a modelar: {len(lista_bancos)}")
    logger.info(f"{'='*55}\n")

    return df_agrupado, lista_bancos, reporte


###############################################################################
# PARTE 4 — Descarga automática de series externas
###############################################################################
def _retry_download(func, max_intentos=3, espera=2):
    """Ejecuta func hasta max_intentos veces con espera exponencial."""
    for intento in range(1, max_intentos + 1):
        try:
            resultado = func()
            if resultado is not None:
                return resultado
        except Exception as e:
            logger.warning(f"  Intento {intento}/{max_intentos} fallido: {e}")
            if intento < max_intentos:
                time.sleep(espera * intento)
    return None


def _descargar_yahoo(ticker, inicio, fin, proxies, nombre):
    """Descarga una serie desde Yahoo Finance."""
    def _descarga():
        import yfinance as yf
        sesion = None
        if proxies:
            import requests
            sesion = requests.Session()
            sesion.proxies.update(proxies)
        t = yf.Ticker(ticker)
        if sesion:
            t._session = sesion
        df = t.history(start=inicio, end=fin, auto_adjust=True)
        if df.empty:
            raise ValueError(f"Sin datos para {ticker}")
        return df["Close"].rename(nombre)

    resultado = _retry_download(_descarga)
    if resultado is None:
        logger.warning(f"  No se pudo descargar {nombre} ({ticker}) desde Yahoo Finance.")
        return pd.Series(dtype=float, name=nombre)
    logger.info(f"  {nombre} descargado: {len(resultado)} observaciones.")
    return resultado


def _descargar_fred(serie_id, inicio, fin, api_key, proxies, nombre):
    """Descarga una serie desde FRED."""
    def _descarga():
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        s = fred.get_series(serie_id, observation_start=inicio, observation_end=fin)
        if s.empty:
            raise ValueError(f"Sin datos para {serie_id}")
        return s.rename(nombre)

    resultado = _retry_download(_descarga)
    if resultado is None:
        logger.warning(f"  No se pudo descargar {nombre} ({serie_id}) desde FRED.")
        return pd.Series(dtype=float, name=nombre)
    logger.info(f"  {nombre} descargado: {len(resultado)} observaciones.")
    return resultado


def _descargar_bcrp(codigo, inicio, fin, proxies, nombre):
    """Descarga una serie desde la API del BCRP."""
    inicio_fmt = pd.to_datetime(inicio).strftime("%d%b%Y").upper()
    fin_fmt = pd.to_datetime(fin).strftime("%d%b%Y").upper()
    url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codigo}/json/{inicio_fmt}/{fin_fmt}"

    def _descarga():
        import requests
        resp = requests.get(url, proxies=proxies if proxies else None, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        periodos = data.get("periods", [])
        if not periodos:
            raise ValueError(f"Sin datos BCRP para {codigo}")
        registros = []
        for p in periodos:
            fecha = pd.to_datetime(p["name"], format="%d.%b.%y", errors="coerce")
            if pd.isna(fecha):
                fecha = pd.to_datetime(p["name"], errors="coerce")
            val = p["values"][0]
            try:
                val = float(str(val).replace(",", "."))
            except (ValueError, TypeError):
                val = np.nan
            registros.append({"fecha": fecha, nombre: val})
        s = pd.DataFrame(registros).set_index("fecha")[nombre].dropna()
        return s

    resultado = _retry_download(_descarga)
    if resultado is None:
        logger.warning(f"  No se pudo descargar {nombre} ({codigo}) desde BCRP.")
        return pd.Series(dtype=float, name=nombre)
    logger.info(f"  {nombre} descargado: {len(resultado)} observaciones.")
    return resultado


def download_external_series(params):
    """
    Descarga todas las series externas y las retorna en un DataFrame
    alineado al calendario hábil peruano.

    Forward fill máximo 3 días para series de menor frecuencia.
    """
    logger.info("PARTE 4: Descargando series externas...")

    inicio = params["fecha_inicio_historico"]
    fin = params["fecha_fin_historico"]
    proxy = params.get("proxy", "")
    proxies = {"http": proxy, "https": proxy} if proxy and "USUARIO" not in proxy else None

    series = {}

    # 4a. Yahoo Finance
    series["VIX"] = _descargar_yahoo(params["ticker_vix"], inicio, fin, proxies, "VIX")
    series["TC_PEN_USD"] = _descargar_yahoo(params["ticker_tc"], inicio, fin, proxies, "TC_PEN_USD")
    series["T10Y"] = _descargar_yahoo(params["ticker_t10y"], inicio, fin, proxies, "T10Y")

    # 4b. FRED
    api_key = params.get("fred_api_key", "")
    if api_key and api_key != "TU_API_KEY":
        series["FED_FUNDS"] = _descargar_fred(
            params["fred_fedfunds"], inicio, fin, api_key, proxies, "FED_FUNDS"
        )
    else:
        logger.warning("  FRED API key no configurada. FED_FUNDS quedará como NaN.")
        series["FED_FUNDS"] = pd.Series(dtype=float, name="FED_FUNDS")

    # 4c. API BCRP
    series["EMBI_PERU"] = _descargar_bcrp(params["bcrp_embi"], inicio, fin, proxies, "EMBI_PERU")
    series["TASA_REF_BCRP"] = _descargar_bcrp(params["bcrp_tasa_ref"], inicio, fin, proxies, "TASA_REF_BCRP")
    series["TC_BCRP"] = _descargar_bcrp(params["bcrp_tc"], inicio, fin, proxies, "TC_BCRP")

    # Alinear al índice de fechas del rango
    idx = pd.bdate_range(start=inicio, end=fin)
    df = pd.DataFrame(index=idx)
    for nombre, s in series.items():
        if not s.empty:
            s_alineada = s.reindex(idx)
            s_alineada = s_alineada.ffill(limit=3)
            df[nombre] = s_alineada
        else:
            df[nombre] = np.nan

    cols_esperadas = ["VIX", "TC_PEN_USD", "T10Y", "FED_FUNDS", "EMBI_PERU", "TASA_REF_BCRP", "TC_BCRP"]
    for c in cols_esperadas:
        if c not in df.columns:
            df[c] = np.nan

    logger.info(f"  DataFrame externo construido: {df.shape}")
    return df


###############################################################################
# PARTE 5 — Construcción de features
###############################################################################

# 5a. Features bancarias
def build_bank_features(df_banco, lags_cortos, lag_semana, lag_mes, ventanas_vol):
    """
    Recibe serie temporal de un banco con columnas R y D.
    Genera features de rezagos, volatilidades, medias móviles y cambios diarios.
    """
    if df_banco.empty or "R" not in df_banco.columns or "D" not in df_banco.columns:
        cols = (
            [f"R_t-{l}" for l in lags_cortos + [lag_semana, lag_mes]]
            + [f"D_t-{l}" for l in lags_cortos + [lag_semana, lag_mes]]
            + [f"sigma_R_{v}d" for v in ventanas_vol]
            + [f"sigma_D_{v}d" for v in ventanas_vol]
            + [f"ma_R_{v}d" for v in ventanas_vol]
            + [f"ma_D_{v}d" for v in ventanas_vol]
            + ["delta_R", "delta_D"]
        )
        return pd.DataFrame(columns=cols)

    df = df_banco[["R", "D"]].copy()
    resultado = pd.DataFrame(index=df.index)

    todos_lags = lags_cortos + [lag_semana, lag_mes]

    for l in todos_lags:
        resultado[f"R_t-{l}"] = df["R"].shift(l)
        resultado[f"D_t-{l}"] = df["D"].shift(l)

    for v in ventanas_vol:
        resultado[f"sigma_R_{v}d"] = df["R"].rolling(v).std()
        resultado[f"sigma_D_{v}d"] = df["D"].rolling(v).std()
        resultado[f"ma_R_{v}d"] = df["R"].rolling(v).mean()
        resultado[f"ma_D_{v}d"] = df["D"].rolling(v).mean()

    resultado["delta_R"] = df["R"].diff(1)
    resultado["delta_D"] = df["D"].diff(1)

    return resultado


# 5b. Features macroeconómicas
def build_macro_features(macro_df):
    """
    Genera features macroeconómicas a partir del DataFrame de series externas.
    """
    if macro_df.empty:
        cols = [
            "VIX", "delta_VIX", "VIX_ma22",
            "TC_PEN_USD", "delta_TC", "tc_vol_5d", "tc_vol_22d",
            "EMBI_PERU", "delta_EMBI",
            "TASA_REF_BCRP", "FED_FUNDS", "diferencial_tasas", "T10Y",
        ]
        return pd.DataFrame(columns=cols)

    df = macro_df.copy()
    resultado = pd.DataFrame(index=df.index)

    resultado["VIX"] = df.get("VIX", np.nan)
    resultado["delta_VIX"] = df.get("VIX", pd.Series(dtype=float)).diff(1)
    resultado["VIX_ma22"] = df.get("VIX", pd.Series(dtype=float)).rolling(22).mean()

    resultado["TC_PEN_USD"] = df.get("TC_PEN_USD", np.nan)
    tc = df.get("TC_PEN_USD", pd.Series(dtype=float))
    resultado["delta_TC"] = tc.diff(1)
    retornos_tc = tc.pct_change()
    resultado["tc_vol_5d"] = retornos_tc.rolling(5).std()
    resultado["tc_vol_22d"] = retornos_tc.rolling(22).std()

    resultado["EMBI_PERU"] = df.get("EMBI_PERU", np.nan)
    resultado["delta_EMBI"] = df.get("EMBI_PERU", pd.Series(dtype=float)).diff(1)

    resultado["TASA_REF_BCRP"] = df.get("TASA_REF_BCRP", np.nan)
    resultado["FED_FUNDS"] = df.get("FED_FUNDS", np.nan)

    tasa_ref = df.get("TASA_REF_BCRP", pd.Series(dtype=float))
    fed_funds = df.get("FED_FUNDS", pd.Series(dtype=float))
    if not tasa_ref.empty and not fed_funds.empty:
        resultado["diferencial_tasas"] = tasa_ref.values - fed_funds.reindex(tasa_ref.index).values
    else:
        resultado["diferencial_tasas"] = np.nan

    resultado["T10Y"] = df.get("T10Y", np.nan)

    return resultado


# 5c. Features estacionales en t+h
def _get_bdays_en_mes(fecha, peru_bday):
    """Retorna todos los días hábiles del mes de la fecha dada."""
    inicio_mes = fecha.replace(day=1)
    if fecha.month == 12:
        fin_mes = fecha.replace(year=fecha.year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        fin_mes = fecha.replace(month=fecha.month + 1, day=1) - pd.Timedelta(days=1)
    return pd.bdate_range(start=inicio_mes, end=fin_mes, freq=peru_bday)


def _get_bdays_en_trim(fecha, peru_bday):
    """Retorna todos los días hábiles del trimestre de la fecha dada."""
    trim = (fecha.month - 1) // 3
    inicio_trim = fecha.replace(month=trim * 3 + 1, day=1)
    if trim == 3:
        fin_trim = fecha.replace(year=fecha.year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        fin_trim = fecha.replace(month=(trim + 1) * 3 + 1, day=1) - pd.Timedelta(days=1)
    return pd.bdate_range(start=inicio_trim, end=fin_trim, freq=peru_bday)


def seasonal_features(fecha, bdays_mes_cache, peru_holidays, fechas_igv, fechas_elecciones):
    """
    Calcula features del CALENDARIO para la fecha futura t+h.
    Estas variables son siempre conocidas porque son determinísticas.
    """
    fecha = pd.Timestamp(fecha)

    # Cache de días hábiles del mes
    clave_mes = (fecha.year, fecha.month)
    if clave_mes not in bdays_mes_cache:
        try:
            from pandas.tseries.offsets import CustomBusinessDay
            peru_bday_local = CustomBusinessDay(holidays=peru_holidays)
        except Exception:
            from pandas.tseries.offsets import BDay
            peru_bday_local = BDay()
        bdays_mes_cache[clave_mes] = _get_bdays_en_mes(fecha, peru_bday_local)

    bdays_mes = bdays_mes_cache[clave_mes]
    total_bdays_mes = len(bdays_mes)

    try:
        pos_en_mes = list(bdays_mes).index(fecha) + 1  # 1-indexed
    except ValueError:
        pos_en_mes = 1

    dias_al_cierre_mes = total_bdays_mes - pos_en_mes
    dias_desde_cierre_mes = pos_en_mes - 1

    # Features de trimestre
    try:
        from pandas.tseries.offsets import CustomBusinessDay
        peru_bday_local = CustomBusinessDay(holidays=peru_holidays)
    except Exception:
        from pandas.tseries.offsets import BDay
        peru_bday_local = BDay()

    bdays_trim = _get_bdays_en_trim(fecha, peru_bday_local)
    try:
        pos_en_trim = list(bdays_trim).index(fecha) + 1
    except ValueError:
        pos_en_trim = 1

    total_bdays_trim = len(bdays_trim)
    is_penult_bday_trim = int(pos_en_trim == total_bdays_trim - 1)
    is_ultimo_bday_trim = int(pos_en_trim == total_bdays_trim)
    is_1er_bday_trim = int(pos_en_trim == 1)
    is_2do_bday_trim = int(pos_en_trim == 2)
    is_3er_bday_trim = int(pos_en_trim == 3)

    # Feriados
    is_pre_feriado = int((fecha + pd.Timedelta(days=1)) in peru_holidays or
                         (fecha + pd.Timedelta(days=2)) in peru_holidays)
    is_post_feriado = int((fecha - pd.Timedelta(days=1)) in peru_holidays or
                          (fecha - pd.Timedelta(days=2)) in peru_holidays)

    # IGV — primer día hábil del mes suele ser pago de IGV
    if fechas_igv:
        fechas_igv_ts = pd.DatetimeIndex(fechas_igv)
        diffs = (fechas_igv_ts - fecha).days
        diffs_pos = diffs[diffs >= 0]
        dias_al_igv = int(diffs_pos.min()) if len(diffs_pos) > 0 else 30
        dias_al_igv = min(dias_al_igv, 30)
        is_igv = int(fecha in fechas_igv_ts)
    else:
        is_igv = 0
        dias_al_igv = 30

    # Elecciones
    if fechas_elecciones:
        fechas_elec_ts = pd.DatetimeIndex(fechas_elecciones)
        is_eleccion = int(fecha in fechas_elec_ts)
        diffs_e = abs((fechas_elec_ts - fecha).days)
        min_diff = diffs_e.min() if len(diffs_e) > 0 else 999
        is_pre_eleccion = int(0 < (pd.DatetimeIndex(fechas_elecciones) - fecha).days.min() <= 7
                              if any((pd.DatetimeIndex(fechas_elecciones) - fecha).days > 0) else False)
        is_post_eleccion = int(0 < (fecha - pd.DatetimeIndex(fechas_elecciones)).days.min() <= 7
                               if any((fecha - pd.DatetimeIndex(fechas_elecciones)).days > 0) else False)
    else:
        is_eleccion = 0
        is_pre_eleccion = 0
        is_post_eleccion = 0

    # Otros indicadores
    dia_semana = fecha.dayofweek  # 0=lunes
    mes = fecha.month
    is_quincena = int(pos_en_mes == 15 or dias_al_cierre_mes == 0)
    is_cierre_encaje = int(dias_al_cierre_mes <= 1)  # Últimos 2 días hábiles del mes
    is_fiestas_patrias = int(mes == 7 and fecha.day in [27, 28, 29])
    is_fin_anio = int(mes == 12 and fecha.day >= 28)

    return {
        "dias_al_cierre_mes": dias_al_cierre_mes,
        "dias_desde_cierre_mes": dias_desde_cierre_mes,
        "pos_en_mes": pos_en_mes,
        "total_bdays_mes": total_bdays_mes,
        "is_penult_bday_trim": is_penult_bday_trim,
        "is_ultimo_bday_trim": is_ultimo_bday_trim,
        "is_1er_bday_trim": is_1er_bday_trim,
        "is_2do_bday_trim": is_2do_bday_trim,
        "is_3er_bday_trim": is_3er_bday_trim,
        "dia_semana": dia_semana,
        "mes": mes,
        "is_quincena": is_quincena,
        "is_cierre_encaje": is_cierre_encaje,
        "is_fiestas_patrias": is_fiestas_patrias,
        "is_fin_anio": is_fin_anio,
        "is_pre_feriado": is_pre_feriado,
        "is_post_feriado": is_post_feriado,
        "is_igv": is_igv,
        "dias_al_igv": dias_al_igv,
        "is_eleccion": is_eleccion,
        "is_pre_eleccion": is_pre_eleccion,
        "is_post_eleccion": is_post_eleccion,
    }


###############################################################################
# PARTE 6 — Construcción de la matriz de features completa
###############################################################################
def build_feature_matrix(
    banco,
    datos_manuales,
    macro_features,
    bank_features,
    peru_bday,
    peru_holidays,
    fechas_igv,
    fechas_elecciones,
    h_min,
    h_max,
):
    """
    Construye el dataset de entrenamiento para un banco.
    Para cada fecha t disponible y cada horizonte h=h_min,...,h_max genera una fila.
    """
    logger.info(f"  Construyendo matriz para banco: {banco}")

    df_bancarios = datos_manuales.get("bancarios", pd.DataFrame())
    df_confirmados = datos_manuales.get("confirmados", pd.DataFrame())

    # Determinar índice temporal de referencia
    if not df_bancarios.empty and f"{banco}_R" in df_bancarios.columns:
        fechas_t = df_bancarios.index
    elif not macro_features.empty:
        fechas_t = macro_features.index
    else:
        logger.warning(f"  Sin fechas de referencia para {banco}. Usando rango histórico.")
        return pd.DataFrame()

    # Preparar datos bancarios del banco
    tiene_bancarios = (
        not df_bancarios.empty
        and f"{banco}_R" in df_bancarios.columns
        and f"{banco}_D" in df_bancarios.columns
    )

    if tiene_bancarios:
        df_banco = df_bancarios[[f"{banco}_R", f"{banco}_D"]].rename(
            columns={f"{banco}_R": "R", f"{banco}_D": "D"}
        )
    else:
        df_banco = pd.DataFrame(columns=["R", "D"], index=fechas_t)

    # Features bancarias para este banco
    bf = bank_features.get(banco, build_bank_features(
        df_banco,
        PARAMS["lags_cortos"],
        PARAMS["lag_semana"],
        PARAMS["lag_mes"],
        PARAMS["ventanas_vol"],
    ))

    # Confirmados
    tiene_confirmados = not df_confirmados.empty and "banco" in df_confirmados.columns
    if tiene_confirmados:
        conf_banco = df_confirmados[df_confirmados["banco"] == banco]
    else:
        conf_banco = pd.DataFrame()

    bdays_mes_cache = {}
    filas = []

    for t in fechas_t:
        # Obtener fechas futuras t+h
        try:
            fechas_futuras = get_future_business_dates(t, h_max, peru_bday)
        except Exception:
            continue

        for idx_h, fecha_th in enumerate(fechas_futuras):
            h = idx_h + 1
            if h < h_min or h > h_max:
                continue

            fila = {
                "fecha_t": t,
                "banco": banco,
                "h": h,
                "log_h": np.log(h),
            }

            # Features bancarias en t
            if not bf.empty and t in bf.index:
                for col in bf.columns:
                    fila[col] = bf.loc[t, col]
            else:
                for col in bf.columns:
                    fila[col] = np.nan

            # Features operativas confirmadas en t
            fila["R_conf_t1"] = np.nan
            fila["R_conf_t2"] = np.nan
            fila["D_conf_t1"] = np.nan
            if not conf_banco.empty:
                if "fecha" in conf_banco.columns:
                    fila_conf = conf_banco[conf_banco["fecha"] == t]
                    if not fila_conf.empty:
                        fila["R_conf_t1"] = fila_conf.iloc[0].get("R_conf_t1", np.nan)
                        fila["R_conf_t2"] = fila_conf.iloc[0].get("R_conf_t2", np.nan)
                        fila["D_conf_t1"] = fila_conf.iloc[0].get("D_conf_t1", np.nan)

            # Features macroeconómicas en t
            if not macro_features.empty and t in macro_features.index:
                for col in macro_features.columns:
                    fila[col] = macro_features.loc[t, col]
            else:
                macro_cols = [
                    "VIX", "delta_VIX", "VIX_ma22",
                    "TC_PEN_USD", "delta_TC", "tc_vol_5d", "tc_vol_22d",
                    "EMBI_PERU", "delta_EMBI",
                    "TASA_REF_BCRP", "FED_FUNDS", "diferencial_tasas", "T10Y",
                ]
                for col in macro_cols:
                    fila[col] = np.nan

            # Features estacionales en t+h
            est = seasonal_features(
                fecha_th, bdays_mes_cache, peru_holidays, fechas_igv, fechas_elecciones
            )
            fila.update(est)

            # Target: D(b, t+h) - R(b, t+h)
            fila["target"] = np.nan
            if tiene_bancarios and fecha_th in df_banco.index:
                r_th = df_banco.loc[fecha_th, "R"]
                d_th = df_banco.loc[fecha_th, "D"]
                if not pd.isna(r_th) and not pd.isna(d_th):
                    fila["target"] = d_th - r_th

            filas.append(fila)

    if not filas:
        logger.warning(f"  Sin filas generadas para banco: {banco}")
        return pd.DataFrame()

    df_matriz = pd.DataFrame(filas)

    # Reporte de NaN
    total = len(df_matriz)
    nan_cols = {
        col: f"{df_matriz[col].isna().sum() / total:.1%}"
        for col in df_matriz.columns
        if df_matriz[col].isna().any()
    }
    logger.info(f"  {banco}: {total:,} filas generadas | columnas con NaN: {len(nan_cols)}")

    return df_matriz


def build_full_matrix(
    params,
    datos_manuales,
    lista_bancos,
    macro_features,
    peru_bday,
    peru_holidays,
    fechas_igv,
    fechas_elecciones,
):
    """
    Llama a build_feature_matrix para cada banco en lista_bancos y concatena resultados.
    Exporta a Excel en ruta_output de PARAMS.
    """
    logger.info("PARTE 6: Construyendo matriz completa de features...")

    df_bancarios = datos_manuales.get("bancarios", pd.DataFrame())

    # Pre-calcular features bancarias por banco
    bank_features_dict = {}
    for banco in lista_bancos:
        if not df_bancarios.empty and f"{banco}_R" in df_bancarios.columns:
            df_banco = df_bancarios[[f"{banco}_R", f"{banco}_D"]].rename(
                columns={f"{banco}_R": "R", f"{banco}_D": "D"}
            )
            bank_features_dict[banco] = build_bank_features(
                df_banco,
                params["lags_cortos"],
                params["lag_semana"],
                params["lag_mes"],
                params["ventanas_vol"],
            )
        else:
            bank_features_dict[banco] = pd.DataFrame()

    matrices = []
    for banco in lista_bancos:
        df_banco_mat = build_feature_matrix(
            banco=banco,
            datos_manuales=datos_manuales,
            macro_features=macro_features,
            bank_features=bank_features_dict,
            peru_bday=peru_bday,
            peru_holidays=peru_holidays,
            fechas_igv=fechas_igv,
            fechas_elecciones=fechas_elecciones,
            h_min=params["h_min"],
            h_max=params["h_max"],
        )
        if not df_banco_mat.empty:
            matrices.append(df_banco_mat)

    if not matrices:
        logger.warning("No se generaron matrices para ningún banco.")
        return pd.DataFrame()

    matriz = pd.concat(matrices, ignore_index=True)

    # Exportar a Excel
    ruta_output = params.get("ruta_output", "matriz_features.xlsx")
    try:
        if ruta_output and "RUTA" not in ruta_output:
            matriz.to_excel(ruta_output, index=False)
            logger.info(f"  Matriz exportada a: {ruta_output}")
        else:
            logger.warning(f"  Ruta de output no configurada. No se exportó a Excel.")
    except Exception as e:
        logger.warning(f"  No se pudo exportar a Excel: {e}")

    # Resumen final
    logger.info(f"\n{'='*60}")
    logger.info("  RESUMEN FINAL DE LA MATRIZ")
    logger.info(f"{'='*60}")
    logger.info(f"  Bancos modelados: {lista_bancos}")
    logger.info(f"  Total filas   : {len(matriz):,}")
    logger.info(f"  Total columnas: {len(matriz.columns)}")

    nan_resumen = {
        col: f"{matriz[col].isna().sum() / len(matriz):.1%}"
        for col in matriz.columns
        if matriz[col].isna().any()
    }
    if nan_resumen:
        logger.info("  Columnas con NaN (% del total):")
        for col, pct in nan_resumen.items():
            logger.info(f"    {col:40s}: {pct}")
    else:
        logger.info("  Sin columnas con NaN.")
    logger.info(f"{'='*60}\n")

    return matriz


###############################################################################
# Datos demo — se usa cuando PARAMS["usar_datos_demo"] = True
###############################################################################
def generate_demo_data(params):
    """
    Genera datos bancarios y macroeconómicos sintéticos para pruebas.
    Simula 5 bancos de distintos tamaños para activar la agrupación automática.
    """
    logger.info("*** MODO DEMO: generando datos sintéticos ***")
    np.random.seed(42)

    fechas = pd.bdate_range(
        start=params["fecha_inicio_historico"],
        end="2024-12-31",
    )
    n = len(fechas)

    # Cinco bancos con escalas muy distintas → banco_4 y banco_5 quedarán en Otros
    escalas = {
        "banco_1": 500,
        "banco_2": 350,
        "banco_3": 200,
        "banco_4": 30,
        "banco_5": 20,
    }

    df_bancarios = pd.DataFrame(index=fechas)
    df_bancarios.index.name = "fecha"
    for banco, escala in escalas.items():
        tendencia = np.linspace(0, escala * 0.1, n)
        ruido_R = np.random.normal(0, escala * 0.15, n)
        ruido_D = np.random.normal(0, escala * 0.15, n)
        df_bancarios[f"{banco}_R"] = escala + tendencia + ruido_R
        df_bancarios[f"{banco}_D"] = escala + tendencia + ruido_D

    # Macro sintética con tendencias plausibles
    df_macro = pd.DataFrame(index=fechas)
    df_macro["VIX"]           = np.clip(20 + np.cumsum(np.random.normal(0, 0.3, n)), 10, 80)
    df_macro["TC_PEN_USD"]    = np.clip(3.3 + np.cumsum(np.random.normal(0, 0.005, n)), 3.0, 4.5)
    df_macro["T10Y"]          = np.clip(3.5 + np.cumsum(np.random.normal(0, 0.02, n)) * 0.05, 1.0, 6.0)
    df_macro["FED_FUNDS"]     = np.clip(1.0 + np.cumsum(np.random.normal(0, 0.01, n)) * 0.03, 0.0, 6.0)
    df_macro["EMBI_PERU"]     = np.clip(130 + np.cumsum(np.random.normal(0, 1.0, n)), 80, 400)
    df_macro["TASA_REF_BCRP"] = np.clip(4.0 + np.cumsum(np.random.normal(0, 0.01, n)) * 0.02, 1.5, 8.0)
    df_macro["TC_BCRP"]       = df_macro["TC_PEN_USD"] + np.random.normal(0, 0.003, n)

    datos_manuales = {
        "bancarios"   : df_bancarios,
        "confirmados" : pd.DataFrame(columns=["banco", "R_conf_t1", "R_conf_t2", "D_conf_t1"]),
        "intervencion": pd.Series(dtype=float),
    }
    return datos_manuales, df_macro


###############################################################################
# Flujo principal
###############################################################################
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  SISTEMA DE PREDICCIÓN DE LIQUIDEZ EN MONEDA EXTRANJERA")
    logger.info("  Construcción de Matriz de Features")
    logger.info(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. Calendario
    peru_bday, peru_holidays, fechas_igv, fechas_elecciones = build_peru_calendar(
        años=PARAMS["años_calendario"],
        ruta_igv=PARAMS["ruta_igv"],
        ruta_elecciones=PARAMS["ruta_elecciones"],
    )

    # 2. Datos manuales + macro
    if PARAMS.get("usar_datos_demo"):
        datos_manuales, macro_raw = generate_demo_data(PARAMS)
    else:
        datos_manuales = load_manual_data(PARAMS)
        macro_raw      = download_external_series(PARAMS)

    # 3. Agrupación de bancos pequeños → Otros_bancos
    df_bancarios_agrupado, lista_bancos, reporte_agrupacion = agrupar_bancos(
        datos_manuales["bancarios"],
        PARAMS["umbral_banco_pequeño_pct"],
        PARAMS["bancos_otros"],
        PARAMS["nombre_otros"],
    )
    datos_manuales["bancarios"] = df_bancarios_agrupado

    if not lista_bancos:
        logger.warning(
            "No hay bancos en lista_bancos. "
            "Ejecutando con banco sintético para validar estructura."
        )
        lista_bancos = ["banco_demo"]

    # 4. Features macroeconómicas
    macro_features = build_macro_features(macro_raw)

    # 5. Matriz completa
    matriz = build_full_matrix(
        params=PARAMS,
        datos_manuales=datos_manuales,
        lista_bancos=lista_bancos,
        macro_features=macro_features,
        peru_bday=peru_bday,
        peru_holidays=peru_holidays,
        fechas_igv=fechas_igv,
        fechas_elecciones=fechas_elecciones,
    )

    if not matriz.empty:
        print(
            f"\nMatriz generada: {matriz.shape[0]:,} filas x "
            f"{matriz.shape[1]} columnas"
        )
    else:
        print("\nMatriz vacía. Verifique la configuración de PARAMS y los archivos de datos.")
