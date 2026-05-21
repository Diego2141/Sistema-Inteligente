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
    "usar_datos_demo": False,

    # Alias de bancos: cambios de nombre históricos que deben unificarse
    # clave = nombre en los datos, valor = nombre canónico
    "alias_bancos": {
        "Continen": "BBVA",   # Banco Continental → BBVA desde 2020
    },

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
    "años_calendario": list(range(2009, 2042)),

    # Rutas archivos manuales
    "ruta_datos_bancarios": r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\Transacciones_BancaLocal.xlsx",
    "ruta_confirmados": r"RUTA\confirmados.xlsx",
    "ruta_intervencion": r"RUTA\intervencion.xlsx",
    # "ruta_igv" eliminado: pagos IGV en soles, no relevante para liquidez ME
    # ruta_elecciones: eliminado — fechas presidenciales hardcodeadas en build_peru_calendar()
    "ruta_aux_xgboost": r"RUTA\Aux_XGBoost.py",
    "ruta_output": r"RUTA\matriz_features.xlsx",
    "ruta_diccionario": r"1. Data/Clean/diccionario_variables.xlsx",

    # Series BCRP descargadas manualmente con Add-In BCRPData
    # Un solo archivo Excel, cada serie en su propia hoja
    "ruta_bcrp": r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\1. Data\Raw\series_bcrp.xlsx",
    "hoja_bcrp_embi":     "EMBIG",      # hoja con PD04709XD (Spread EMBIG Perú)
    "hoja_bcrp_tasa_ref": "TasaPM",     # hoja con PD12301MD (Tasa de Referencia)
    "hoja_bcrp_tc_compra": "TC Compra", # hoja con TC compra BCRP
    "hoja_bcrp_tc_venta":  "TC Venta",  # hoja con TC venta BCRP

    # APIs externas
    "fred_api_key": "96fa168ee9a9a4c1fcf323983db5ba64",
    "proxy": "http://2577:fgh1103.@bcrproxy:8080",

    # Códigos BCRP (solo como referencia; la lectura es desde archivo)
    "bcrp_embi": "PD04709XD",      # Spread EMBIG Perú (pbs) — diaria
    "bcrp_tasa_ref": "PD12301MD",  # Tasa de Referencia Política Monetaria — diaria
    # TC se lee de BCRP (promedio compra/venta) — Yahoo Finance tenía outliers

    # Tickers Yahoo Finance
    "ticker_vix": "^VIX",
    "ticker_tc": "PEN=X",          # Tipo de cambio USD/PEN (venta)
    "ticker_t10y": "^TNX",

    # Código FRED
    "fred_fedfunds": "FEDFUNDS",
}


###############################################################################
# PARTE 1 — Calendario hábil peruano
###############################################################################
def build_peru_calendar(años, ruta_elecciones=None):
    """
    Retorna (peru_bday, peru_holidays, fechas_elecciones).

    Calendario hábil conjunto PE + US: excluye feriados de ambos países.
    Elecciones presidenciales hardcodeadas 2000-2041 (no requieren archivo externo).
    """
    logger.info("PARTE 1: Construyendo calendario hábil peruano...")

    try:
        import holidays
        from pandas.tseries.offsets import CustomBusinessDay

        peru_hols = holidays.Peru(years=años)
        us_hols   = holidays.UnitedStates(years=años)

        # Unión de feriados PE + US: un día no hábil si lo es en cualquiera de los dos países
        joint_hols_keys = set(peru_hols.keys()) | set(us_hols.keys())
        peru_holidays   = pd.to_datetime(sorted(joint_hols_keys))
        peru_bday       = CustomBusinessDay(holidays=peru_holidays)

        logger.info(
            f"  Feriados cargados: {len(peru_hols)} PE + {len(us_hols)} US "
            f"= {len(peru_holidays)} únicos en {min(años)}-{max(años)}"
        )
    except ImportError:
        logger.warning("  Librería 'holidays' no instalada. Usando calendario sin feriados.")
        from pandas.tseries.offsets import BDay
        peru_bday = BDay()
        peru_holidays = pd.DatetimeIndex([])

    # Elecciones presidenciales peruanas hardcodeadas 2000-2041
    # Fechas desde 2031 son estimadas (2do domingo de abril / 1er domingo de junio)
    ULTIMA_FECHA_CONFIRMADA = pd.Timestamp("2026-06-07")
    ULTIMO_ANIO_HORIZONTE   = 2041

    _elecciones_raw = [
        # ── Históricas confirmadas ──────────────────────────────
        ("2000-04-09", "1ra vuelta"),  ("2000-05-28", "2da vuelta"),
        ("2001-04-08", "1ra vuelta"),  ("2001-06-03", "2da vuelta"),
        ("2006-04-09", "1ra vuelta"),  ("2006-06-04", "2da vuelta"),
        ("2011-04-10", "1ra vuelta"),  ("2011-06-05", "2da vuelta"),
        ("2016-04-10", "1ra vuelta"),  ("2016-06-05", "2da vuelta"),
        ("2021-04-11", "1ra vuelta"),  ("2021-06-06", "2da vuelta"),
        # ── Programadas / confirmadas ───────────────────────────
        ("2026-04-12", "1ra vuelta"),  ("2026-06-07", "2da vuelta"),
        # ── Estimadas (cada 5 años, 2do domingo abril / ~7 sem después) ──
        ("2031-04-13", "1ra vuelta"),  ("2031-06-08", "2da vuelta"),
        ("2036-04-12", "1ra vuelta"),  ("2036-06-07", "2da vuelta"),
        ("2041-04-14", "1ra vuelta"),  ("2041-06-08", "2da vuelta"),
    ]

    fechas_elecciones = [pd.Timestamp(f) for f, _ in _elecciones_raw]
    logger.info(
        f"  Elecciones presidenciales cargadas: {len(fechas_elecciones)} fechas "
        f"(2000-{ULTIMO_ANIO_HORIZONTE}, estimadas desde 2031)"
    )

    # Avisar cuando el horizonte de predicción se acerca al límite estimado
    fecha_hoy = pd.Timestamp.today().normalize()
    dias_al_limite = (pd.Timestamp(f"{ULTIMO_ANIO_HORIZONTE}-12-31") - fecha_hoy).days
    if dias_al_limite < 365 * 3:
        logger.warning(
            f"  ⚠ HORIZONTE ELECTORAL: quedan ~{dias_al_limite // 365} años hasta el límite "
            f"estimado ({ULTIMO_ANIO_HORIZONTE}). Considera extender las fechas en el código."
        )
    primera_estimada = pd.Timestamp("2031-04-13")
    dias_a_estimada = (primera_estimada - fecha_hoy).days
    if dias_a_estimada < 365 * 2:
        logger.warning(
            f"  ⚠ ELECCIONES ESTIMADAS: en ~{dias_a_estimada // 30} meses se entra a fechas "
            f"estimadas (desde {primera_estimada.date()}). Verificar con JNE."
        )

    logger.info("  Calendario construido correctamente.")
    return peru_bday, peru_holidays, fechas_elecciones


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

    # Datos bancarios — formato Transacciones_BancaLocal.xlsx
    # Columnas: Broker (banco), Fecha Valor (fecha), Delivery Principal Usd (monto)
    # Monto < 0 → Retiro (R) | Monto > 0 → Depósito (D)
    try:
        df_raw = pd.read_excel(params["ruta_datos_bancarios"])
        if df_raw.empty:
            logger.warning(f"  Archivo bancarios vacío: {params['ruta_datos_bancarios']}")
        else:
            df_raw["Fecha Valor"] = pd.to_datetime(df_raw["Fecha Valor"])
            df_raw["monto"] = pd.to_numeric(df_raw["Delivery Principal Usd"], errors="coerce")

            # Separar retiros y depósitos
            df_raw["R"] = df_raw["monto"].clip(upper=0).abs()  # negativos → positivos
            df_raw["D"] = df_raw["monto"].clip(lower=0)        # positivos

            # Agregar por banco y fecha
            df_banco = (
                df_raw.groupby(["Broker", "Fecha Valor"])[["R", "D"]]
                .sum()
                .reset_index()
                .rename(columns={"Broker": "banco", "Fecha Valor": "fecha"})
            )
            df_banco["fecha"] = pd.to_datetime(df_banco["fecha"])

            # Unificar alias históricos (ej. Continen → BBVA)
            alias = params.get("alias_bancos", {})
            if alias:
                df_banco["banco"] = df_banco["banco"].replace(alias)
                # Re-agregar por si el mismo banco tiene filas con ambos nombres en la misma fecha
                df_banco = (
                    df_banco.groupby(["banco", "fecha"])[["R", "D"]]
                    .sum()
                    .reset_index()
                )
                for viejo, nuevo in alias.items():
                    if viejo in df_banco["banco"].values or nuevo in df_banco["banco"].values:
                        logger.info(f"  Alias aplicado: '{viejo}' → '{nuevo}'")

            df_banco = df_banco.sort_values(["banco", "fecha"])
            resultado["bancarios"] = df_banco
            n_bancos = df_banco["banco"].nunique()
            f_min = df_banco["fecha"].min().date()
            f_max = df_banco["fecha"].max().date()
            logger.info(
                f"  Datos bancarios cargados: {len(df_banco):,} filas | "
                f"{n_bancos} bancos | {f_min} → {f_max}"
            )
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
    """
    Descarga una serie desde Yahoo Finance via requests directo con headers de navegador.
    Funciona detrás de proxies corporativos que interceptan SSL.
    """
    import urllib3
    urllib3.disable_warnings()

    ticker_encoded = ticker.replace("^", "%5E")
    inicio_ts = int(pd.Timestamp(inicio).timestamp())
    fin_ts    = int(pd.Timestamp(fin).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_encoded}"
        f"?interval=1d&period1={inicio_ts}&period2={fin_ts}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com",
    }

    def _descarga():
        import requests as req
        r = req.get(
            url,
            headers=headers,
            proxies=proxies if proxies else None,
            verify=False,
            timeout=30,
        )
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code} para {ticker}")
        data   = r.json()
        result = data["chart"]["result"]
        if not result:
            raise ValueError(f"Sin datos en respuesta para {ticker}")
        timestamps = result[0]["timestamp"]
        closes     = result[0]["indicators"]["quote"][0]["close"]
        s = pd.Series(
            closes,
            index=pd.to_datetime(timestamps, unit="s").tz_localize(None),
            name=nombre,
            dtype=float,
        )
        s.index = s.index.normalize()
        if s.dropna().empty:
            raise ValueError(f"Serie vacía para {ticker}")
        return s

    resultado = _retry_download(_descarga)
    if resultado is None:
        logger.warning(f"  No se pudo descargar {nombre} ({ticker}) desde Yahoo Finance.")
        return pd.Series(dtype=float, name=nombre)
    logger.info(f"  {nombre} descargado: {len(resultado)} observaciones.")
    return resultado


def _descargar_fred(serie_id, inicio, fin, api_key, proxies, nombre):
    """Descarga una serie desde la API REST de FRED (sin fredapi, solo requests)."""
    import urllib3
    urllib3.disable_warnings()

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={serie_id}"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&observation_start={pd.Timestamp(inicio).strftime('%Y-%m-%d')}"
        f"&observation_end={pd.Timestamp(fin).strftime('%Y-%m-%d')}"
    )

    def _descarga():
        import requests
        r = requests.get(
            url,
            proxies=proxies if proxies else None,
            verify=False,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            raise ValueError(f"FRED HTTP {r.status_code}: {r.text[:200]}")
        obs = r.json().get("observations", [])
        if not obs:
            raise ValueError(f"Sin observaciones para {serie_id}")
        registros = []
        for o in obs:
            try:
                val = float(o["value"])
                registros.append({"fecha": pd.Timestamp(o["date"]), nombre: val})
            except (ValueError, TypeError):
                pass  # "." = dato faltante en FRED
        s = pd.DataFrame(registros).set_index("fecha")[nombre]
        return s

    resultado = _retry_download(_descarga)
    if resultado is None:
        logger.warning(f"  No se pudo descargar {nombre} ({serie_id}) desde FRED.")
        return pd.Series(dtype=float, name=nombre)
    logger.info(f"  {nombre} descargado: {len(resultado)} observaciones.")
    return resultado


def _leer_bcrp_excel(ruta, nombre, hoja=0):
    """
    Lee una serie BCRP desde una hoja de Excel descargado con el Add-In BCRPData.

    Formato esperado del Add-In:
        Fila 1: código de serie
        Fila 2: Descripción
        Fila 3: Frecuencia
        Fila 4: Periodo
        Fila 5: vacía
        Fila 6: encabezados  Date | Valores
        Fila 7+: datos       fecha | valor

    hoja: nombre o índice de la hoja dentro del Excel (default=0, primera hoja).
    Si el archivo no existe devuelve Series vacía con advertencia.
    """
    if not ruta or not os.path.exists(ruta):
        logger.warning(
            f"  Archivo BCRP '{nombre}' no encontrado en: {ruta}\n"
            f"  → Descargarlo con Add-In BCRPData y guardarlo en esa ruta."
        )
        return pd.Series(dtype=float, name=nombre)

    try:
        # header=None para leer desde fila 1 sin asumir encabezados
        raw = pd.read_excel(ruta, sheet_name=hoja, header=None)

        # Buscar la fila de encabezado "Date" / "Valores"
        header_row = None
        for i, row in raw.iterrows():
            vals = [str(v).strip().lower() for v in row if pd.notna(v)]
            if any(v in ("date", "fecha") for v in vals):
                header_row = i
                break

        if header_row is None:
            header_row = 5  # fallback: fila 6

        df = pd.read_excel(ruta, sheet_name=hoja, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        # Identificar columnas de fecha y valor
        col_fecha = next((c for c in df.columns if c.lower() in ("date", "fecha")), df.columns[0])
        col_valor = next((c for c in df.columns if c.lower() in ("valores", "value", "values")), df.columns[1])

        df = df[[col_fecha, col_valor]].copy()
        df.columns = ["fecha", nombre]

        # El Add-In BCRPData usa meses en español: Set=Sep, Ene=Jan, etc.
        _meses_es = {
            "Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr",
            "May": "May", "Jun": "Jun", "Jul": "Jul", "Ago": "Aug",
            "Set": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec",
        }
        def _parse_fecha_bcrp(s):
            s = str(s).strip()
            for es, en in _meses_es.items():
                s = s.replace(es, en)
            return pd.to_datetime(s, errors="coerce", dayfirst=True)

        df["fecha"] = df["fecha"].apply(_parse_fecha_bcrp)
        df[nombre] = pd.to_numeric(
            df[nombre].astype(str).str.replace(",", "."), errors="coerce"
        )
        df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()

        # Advertir si los datos tienen más de 7 días de antigüedad
        ultima = df.index.max()
        dias_atras = (pd.Timestamp.today() - ultima).days
        if dias_atras > 7:
            logger.warning(
                f"  {nombre}: última fecha {ultima.date()} ({dias_atras} días atrás). "
                f"Considera actualizar el archivo."
            )

        logger.info(f"  {nombre} leído desde archivo: {len(df)} obs, hasta {ultima.date()}")
        return df[nombre]

    except Exception as e:
        logger.warning(f"  Error leyendo {nombre} desde {ruta}: {e}")
        return pd.Series(dtype=float, name=nombre)


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
    series["VIX"]  = _descargar_yahoo(params["ticker_vix"],  inicio, fin, proxies, "VIX")
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

    # 4c. Series BCRP — leídas desde Excel descargado manualmente con Add-In BCRPData
    ruta_bcrp = params["ruta_bcrp"]
    series["EMBI_PERU"]     = _leer_bcrp_excel(ruta_bcrp, "EMBI_PERU",     hoja=params["hoja_bcrp_embi"])
    series["TASA_REF_BCRP"] = _leer_bcrp_excel(ruta_bcrp, "TASA_REF_BCRP", hoja=params["hoja_bcrp_tasa_ref"])

    # TC_PEN_USD = promedio(TC Compra, TC Venta) del BCRP — reemplaza Yahoo Finance (tenía outliers)
    tc_compra = _leer_bcrp_excel(ruta_bcrp, "TC_COMPRA", hoja=params["hoja_bcrp_tc_compra"])
    tc_venta  = _leer_bcrp_excel(ruta_bcrp, "TC_VENTA",  hoja=params["hoja_bcrp_tc_venta"])
    if not tc_compra.empty and not tc_venta.empty:
        tc_df = pd.concat([tc_compra.rename("compra"), tc_venta.rename("venta")], axis=1)
        series["TC_PEN_USD"] = tc_df.mean(axis=1).rename("TC_PEN_USD")
        logger.info("  TC_PEN_USD: promedio BCRP compra/venta cargado.")
    elif not tc_venta.empty:
        series["TC_PEN_USD"] = tc_venta.rename("TC_PEN_USD")
        logger.warning("  TC_PEN_USD: solo TC Venta disponible (sin TC Compra).")
    elif not tc_compra.empty:
        series["TC_PEN_USD"] = tc_compra.rename("TC_PEN_USD")
        logger.warning("  TC_PEN_USD: solo TC Compra disponible (sin TC Venta).")
    else:
        logger.warning("  TC_PEN_USD: no se encontraron hojas TC Compra/Venta en series_bcrp.xlsx.")
        series["TC_PEN_USD"] = pd.Series(dtype=float, name="TC_PEN_USD")

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

    cols_esperadas = ["VIX", "TC_PEN_USD", "T10Y", "FED_FUNDS", "EMBI_PERU", "TASA_REF_BCRP"]
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
            ["R_t0", "D_t0"]
            + [f"R_t-{l}" for l in lags_cortos + [lag_semana, lag_mes]]
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

    # Valores de hoy: disponibles en t por aviso anticipado
    # R avisado 2 días antes → R(t) conocido en t; D avisado 1 día antes → D(t) conocido en t
    resultado["R_t0"] = df["R"]
    resultado["D_t0"] = df["D"]

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


def seasonal_features(fecha, bdays_mes_cache, peru_holidays, fechas_elecciones):
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

    # Elecciones — is_eleccion eliminado: elecciones peruanas siempre en domingo (día no hábil)
    if fechas_elecciones:
        is_pre_eleccion = int(0 < (pd.DatetimeIndex(fechas_elecciones) - fecha).days.min() <= 7
                              if any((pd.DatetimeIndex(fechas_elecciones) - fecha).days > 0) else False)
        is_post_eleccion = int(0 < (fecha - pd.DatetimeIndex(fechas_elecciones)).days.min() <= 7
                               if any((fecha - pd.DatetimeIndex(fechas_elecciones)).days > 0) else False)
    else:
        is_pre_eleccion = 0
        is_post_eleccion = 0

    # Otros indicadores
    dia_semana = fecha.dayofweek  # 0=lunes
    mes = fecha.month
    is_quincena = int(pos_en_mes == 15 or dias_al_cierre_mes == 0)
    is_cierre_encaje = int(dias_al_cierre_mes <= 1)  # Últimos 2 días hábiles del mes
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
        "is_fin_anio": is_fin_anio,
        "is_pre_feriado": is_pre_feriado,
        "is_post_feriado": is_post_feriado,
        "is_pre_eleccion": is_pre_eleccion,
        "is_post_eleccion": is_post_eleccion,
    }


###############################################################################
# PARTE 6 — Construcción de la matriz de features completa
###############################################################################
def _build_seasonal_table(fechas_unicas, peru_holidays, fechas_elecciones, peru_bday):
    """
    Calcula features estacionales para un conjunto de fechas únicas (t+h).
    Retorna DataFrame indexado por fecha.
    """
    # Pre-calcular días hábiles por mes y trimestre una sola vez
    meses_unicos = pd.PeriodIndex(fechas_unicas, freq="M").unique()
    trims_unicos = pd.PeriodIndex(fechas_unicas, freq="Q").unique()

    # Mapa: fecha → días hábiles de su mes
    bdays_por_mes = {}
    for m in meses_unicos:
        inicio = m.start_time
        fin    = m.end_time
        bdays_por_mes[m] = pd.bdate_range(start=inicio, end=fin, freq=peru_bday)

    # Mapa: fecha → días hábiles de su trimestre
    bdays_por_trim = {}
    for q in trims_unicos:
        inicio = q.start_time
        fin    = q.end_time
        bdays_por_trim[q] = pd.bdate_range(start=inicio, end=fin, freq=peru_bday)

    # Sets para lookups rápidos
    hols_set  = set(peru_holidays)
    elec_set  = set(pd.DatetimeIndex(fechas_elecciones)) if fechas_elecciones else set()
    elec_arr  = np.array(sorted(elec_set), dtype="datetime64[D]") if elec_set else np.array([], dtype="datetime64[D]")

    registros = []
    for fecha in fechas_unicas:
        ts = pd.Timestamp(fecha)
        m  = ts.to_period("M")
        q  = ts.to_period("Q")

        bd_mes  = bdays_por_mes.get(m,  pd.DatetimeIndex([]))
        bd_trim = bdays_por_trim.get(q, pd.DatetimeIndex([]))
        bd_mes_list  = list(bd_mes)
        bd_trim_list = list(bd_trim)

        total_bdays_mes  = len(bd_mes_list)
        total_bdays_trim = len(bd_trim_list)

        try:
            pos_en_mes  = bd_mes_list.index(ts) + 1
        except ValueError:
            pos_en_mes  = 1
        try:
            pos_en_trim = bd_trim_list.index(ts) + 1
        except ValueError:
            pos_en_trim = 1

        dias_al_cierre_mes    = total_bdays_mes - pos_en_mes
        dias_desde_cierre_mes = pos_en_mes - 1

        is_penult_bday_trim = int(pos_en_trim == total_bdays_trim - 1)
        is_ultimo_bday_trim = int(pos_en_trim == total_bdays_trim)
        is_1er_bday_trim    = int(pos_en_trim == 1)
        is_2do_bday_trim    = int(pos_en_trim == 2)
        is_3er_bday_trim    = int(pos_en_trim == 3)

        # Feriados (busca en ventana ±2 días calendario)
        is_pre_feriado  = int(any((ts + pd.Timedelta(days=d)) in hols_set for d in [1, 2]))
        is_post_feriado = int(any((ts - pd.Timedelta(days=d)) in hols_set for d in [1, 2]))

        # Elecciones
        if len(elec_arr) > 0:
            ts_d  = np.datetime64(ts, "D")
            diffs = (elec_arr - ts_d).astype(int)
            futuros_e        = diffs[diffs > 0]
            pasados_e        = (-diffs)[diffs < 0]
            is_pre_eleccion  = int(len(futuros_e) > 0 and futuros_e.min() <= 7)
            is_post_eleccion = int(len(pasados_e) > 0 and pasados_e.min() <= 7)
        else:
            is_pre_eleccion = is_post_eleccion = 0

        mes = ts.month
        registros.append({
            "fecha_th"             : ts,
            "dias_al_cierre_mes"   : dias_al_cierre_mes,
            "dias_desde_cierre_mes": dias_desde_cierre_mes,
            "pos_en_mes"           : pos_en_mes,
            "total_bdays_mes"      : total_bdays_mes,
            "is_penult_bday_trim"  : is_penult_bday_trim,
            "is_ultimo_bday_trim"  : is_ultimo_bday_trim,
            "is_1er_bday_trim"     : is_1er_bday_trim,
            "is_2do_bday_trim"     : is_2do_bday_trim,
            "is_3er_bday_trim"     : is_3er_bday_trim,
            "dia_semana"           : ts.dayofweek,
            "mes"                  : mes,
            "is_quincena"          : int(pos_en_mes == 15 or dias_al_cierre_mes == 0),
            "is_cierre_encaje"     : int(dias_al_cierre_mes <= 1),
            "is_fin_anio"          : int(mes == 12 and ts.day >= 28),
            "is_pre_feriado"       : is_pre_feriado,
            "is_post_feriado"      : is_post_feriado,
            "is_pre_eleccion"      : is_pre_eleccion,
            "is_post_eleccion"     : is_post_eleccion,
        })

    return pd.DataFrame(registros).set_index("fecha_th")


def build_feature_matrix(
    banco,
    datos_manuales,
    macro_features,
    bank_features,
    peru_bday,
    peru_holidays,
    fechas_elecciones,
    h_min,
    h_max,
):
    """
    Construye el dataset de entrenamiento para un banco de forma vectorizada.
    """
    logger.info(f"  Construyendo matriz para banco: {banco}")

    df_bancarios  = datos_manuales.get("bancarios",  pd.DataFrame())
    df_confirmados = datos_manuales.get("confirmados", pd.DataFrame())

    # Índice temporal de referencia
    if not df_bancarios.empty and f"{banco}_R" in df_bancarios.columns:
        fechas_t = df_bancarios.index
    elif not macro_features.empty:
        fechas_t = macro_features.index
    else:
        logger.warning(f"  Sin fechas de referencia para {banco}.")
        return pd.DataFrame()

    fechas_t = pd.DatetimeIndex(fechas_t)

    # ── 1. Grid (fecha_t × h) ────────────────────────────────────────────────
    hs = np.arange(h_min, h_max + 1)
    grid = pd.MultiIndex.from_product([fechas_t, hs], names=["fecha_t", "h"])
    df = pd.DataFrame(index=grid).reset_index()
    df["log_h"] = np.log(df["h"])
    df["banco"] = banco

    # Calcular fecha_th vectorialmente: aplicar el offset h veces
    # Construimos un mapa fecha_t → array de fechas futuras
    logger.info(f"    Calculando fechas futuras ({len(fechas_t)} × {len(hs)})...")
    th_map = {}
    for t in fechas_t:
        th_map[t] = list(get_future_business_dates(t, h_max, peru_bday))

    def get_th(row):
        fechas = th_map.get(row["fecha_t"], [])
        idx = row["h"] - 1
        return fechas[idx] if idx < len(fechas) else pd.NaT

    df["fecha_th"] = df.apply(get_th, axis=1)
    df = df.dropna(subset=["fecha_th"])

    # ── 2. Features bancarias (merge por fecha_t) ────────────────────────────
    bf = bank_features.get(banco, pd.DataFrame())
    if not bf.empty:
        df = df.merge(bf.add_prefix(""), left_on="fecha_t", right_index=True, how="left")

    # ── 3. Datos bancarios → target ──────────────────────────────────────────
    tiene_bancarios = (
        not df_bancarios.empty
        and f"{banco}_R" in df_bancarios.columns
        and f"{banco}_D" in df_bancarios.columns
    )
    if tiene_bancarios:
        df_banco = df_bancarios[[f"{banco}_R", f"{banco}_D"]].copy()
        df_banco.columns = ["R_th", "D_th"]
        df_banco["target"] = df_banco["D_th"] - df_banco["R_th"]
        df = df.merge(df_banco[["target"]], left_on="fecha_th", right_index=True, how="left")
    else:
        df["target"] = np.nan

    # ── 4. Confirmados (merge por fecha_t) ───────────────────────────────────
    for col in ["R_conf_t1", "R_conf_t2", "D_conf_t1"]:
        df[col] = np.nan
    if not df_confirmados.empty and "banco" in df_confirmados.columns and "fecha" in df_confirmados.columns:
        conf = df_confirmados[df_confirmados["banco"] == banco].set_index("fecha")
        for col in ["R_conf_t1", "R_conf_t2", "D_conf_t1"]:
            if col in conf.columns:
                df[col] = df["fecha_t"].map(conf[col])

    # ── 5. Macro (merge por fecha_t) ─────────────────────────────────────────
    if not macro_features.empty:
        df = df.merge(macro_features, left_on="fecha_t", right_index=True, how="left")
    else:
        for col in ["VIX","delta_VIX","VIX_ma22","TC_PEN_USD","delta_TC",
                    "tc_vol_5d","tc_vol_22d","EMBI_PERU","delta_EMBI",
                    "TASA_REF_BCRP","FED_FUNDS","diferencial_tasas","T10Y"]:
            df[col] = np.nan

    # ── 6. Estacionales (calculadas una vez por fecha_th única) ──────────────
    fechas_th_unicas = pd.DatetimeIndex(df["fecha_th"].unique())
    logger.info(f"    Calculando estacionales para {len(fechas_th_unicas):,} fechas únicas...")
    df_seasonal = _build_seasonal_table(
        fechas_th_unicas, peru_holidays, fechas_elecciones, peru_bday
    )
    df = df.merge(df_seasonal, left_on="fecha_th", right_index=True, how="left")

    # ── Reordenar columnas ───────────────────────────────────────────────────
    cols_id = ["fecha_t", "banco", "h", "log_h"]
    cols_resto = [c for c in df.columns if c not in cols_id + ["fecha_th"]]
    df = df[cols_id + cols_resto].drop(columns=["fecha_th"], errors="ignore")

    # Reporte NaN
    total = len(df)
    nan_cols = {c: f"{df[c].isna().sum()/total:.1%}" for c in df.columns if df[c].isna().any()}
    logger.info(f"  {banco}: {total:,} filas | columnas con NaN: {len(nan_cols)}")

    return df


def build_full_matrix(
    params,
    datos_manuales,
    lista_bancos,
    macro_features,
    peru_bday,
    peru_holidays,
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
# Diccionario de variables
###############################################################################
def build_data_dictionary(params):
    """
    Retorna un DataFrame con todas las variables de la matriz,
    su fuente, descripción y rezago/horizonte.
    """
    lags_cortos  = params["lags_cortos"]
    lag_semana   = params["lag_semana"]
    lag_mes      = params["lag_mes"]
    ventanas_vol = params["ventanas_vol"]

    registros = []

    def add(variable, fuente, descripcion, rezago_dias=None, horizonte=None):
        registros.append({
            "variable"    : variable,
            "fuente"      : fuente,
            "descripcion" : descripcion,
            "rezago_dias" : rezago_dias,
            "horizonte"   : horizonte,
        })

    # ── Identificadores ──────────────────────────────────────────────────────
    add("fecha_t",  "Sistema", "Fecha de origen de la predicción",       None, None)
    add("banco",    "Sistema", "Identificador del banco",                 None, None)
    add("h",        "Sistema", "Horizonte de predicción en días hábiles", None, "t+h")
    add("log_h",    "Sistema", "Logaritmo natural del horizonte h",       None, "t+h")

    # ── Features bancarias ────────────────────────────────────────────────────
    # Estructura de información disponible en t:
    #   R (retiros, aviso 2 días hábiles antes):
    #     t-2 → R(t) disponible como R_t0
    #     t-1 → R(t+1) disponible como R_conf_t1
    #     t   → R(t+2) disponible como R_conf_t2
    #   D (depósitos, aviso 1 día hábil antes):
    #     t-1 → D(t) disponible como D_t0
    #     t   → D(t+1) disponible como D_conf_t1

    # Valores de hoy: conocidos por aviso anticipado
    add("R_t0", "Datos bancarios", "Retiro realizado en t — conocido hoy (avisado en t-2)", 0, None)
    add("D_t0", "Datos bancarios", "Depósito realizado en t — conocido hoy (avisado en t-1)", 0, None)

    # Rezagos históricos
    todos_lags = lags_cortos + [lag_semana, lag_mes]
    for l in todos_lags:
        add(f"R_t-{l}", "Datos bancarios", f"Retiro del banco en t-{l}", l, None)
        add(f"D_t-{l}", "Datos bancarios", f"Depósito del banco en t-{l}", l, None)

    for v in ventanas_vol:
        add(f"sigma_R_{v}d", "Datos bancarios", f"Desv. estándar rolling {v}d de retiros (incluye R_t0)",    None, None)
        add(f"sigma_D_{v}d", "Datos bancarios", f"Desv. estándar rolling {v}d de depósitos (incluye D_t0)",  None, None)
        add(f"ma_R_{v}d",    "Datos bancarios", f"Media móvil {v}d de retiros (incluye R_t0)",                None, None)
        add(f"ma_D_{v}d",    "Datos bancarios", f"Media móvil {v}d de depósitos (incluye D_t0)",              None, None)

    add("delta_R", "Datos bancarios", "Variación diaria de retiros: R_t0 - R(t-1)",   0, None)
    add("delta_D", "Datos bancarios", "Variación diaria de depósitos: D_t0 - D(t-1)", 0, None)

    # ── Confirmados futuros (notificados formalmente por los bancos) ──────────
    add("R_conf_t1", "Confirmados operativos", "Retiro confirmado para t+1 (notificado en t-1, aviso 2d)", -1, "t+1")
    add("R_conf_t2", "Confirmados operativos", "Retiro confirmado para t+2 (notificado hoy, aviso 2d)",     0, "t+2")
    add("D_conf_t1", "Confirmados operativos", "Depósito confirmado para t+1 (notificado hoy, aviso 1d)",   0, "t+1")

    # ── Features macroeconómicas (observadas en t) ───────────────────────────
    add("VIX",              "Yahoo Finance",  "Índice de volatilidad implícita S&P 500",          0, None)
    add("delta_VIX",        "Yahoo Finance",  "Variación diaria del VIX",                         1, None)
    add("VIX_ma22",         "Yahoo Finance",  "Media móvil 22d del VIX",                          None, None)
    add("TC_PEN_USD",       "BCRP Add-In",    "Tipo de cambio PEN/USD (promedio compra/venta)",   0, None)
    add("delta_TC",         "BCRP Add-In",    "Variación diaria del tipo de cambio",              1, None)
    add("tc_vol_5d",        "BCRP Add-In",    "Volatilidad rolling 5d de retornos del TC",        None, None)
    add("tc_vol_22d",       "BCRP Add-In",    "Volatilidad rolling 22d de retornos del TC",       None, None)
    add("EMBI_PERU",        "BCRP Add-In",    "EMBI Perú (riesgo país)",                          0, None)
    add("delta_EMBI",       "BCRP Add-In",    "Variación diaria del EMBI Perú",                   1, None)
    add("TASA_REF_BCRP",    "BCRP Add-In",    "Tasa de referencia del BCRP",                      0, None)
    add("FED_FUNDS",        "FRED API",       "Tasa de política monetaria de la Fed",             0, None)
    add("diferencial_tasas","Calculado",      "TASA_REF_BCRP - FED_FUNDS",                        0, None)
    add("T10Y",             "Yahoo Finance",  "Rendimiento del bono del Tesoro EE.UU. a 10 años", 0, None)

    # ── Features estacionales (en t+h — fecha futura, siempre conocidas) ─────
    add("dias_al_cierre_mes",    "Calendario", "Días hábiles restantes hasta fin de mes en t+h",          None, "t+h")
    add("dias_desde_cierre_mes", "Calendario", "Días hábiles transcurridos desde inicio de mes en t+h",   None, "t+h")
    add("pos_en_mes",            "Calendario", "Posición del día hábil dentro del mes (1=primero)",        None, "t+h")
    add("total_bdays_mes",       "Calendario", "Total de días hábiles del mes de t+h",                     None, "t+h")
    add("is_penult_bday_trim",   "Calendario", "1 si t+h es penúltimo día hábil del trimestre",            None, "t+h")
    add("is_ultimo_bday_trim",   "Calendario", "1 si t+h es último día hábil del trimestre",               None, "t+h")
    add("is_1er_bday_trim",      "Calendario", "1 si t+h es primer día hábil del trimestre",               None, "t+h")
    add("is_2do_bday_trim",      "Calendario", "1 si t+h es segundo día hábil del trimestre",              None, "t+h")
    add("is_3er_bday_trim",      "Calendario", "1 si t+h es tercer día hábil del trimestre",               None, "t+h")
    add("dia_semana",            "Calendario", "Día de la semana (0=lunes, 4=viernes)",                    None, "t+h")
    add("mes",                   "Calendario", "Mes del año (1–12)",                                       None, "t+h")
    add("is_quincena",           "Calendario", "1 si t+h es día 15 o último hábil del mes",                None, "t+h")
    add("is_cierre_encaje",      "Calendario", "1 si t+h está en los últimos 2 días hábiles del mes",      None, "t+h")
    add("is_fin_anio",           "Calendario", "1 si t+h es 28–31 de diciembre",                           None, "t+h")
    add("is_pre_feriado",        "Calendario / holidays.Peru + holidays.US", "1 si el día siguiente a t+h es feriado PE o US",  None, "t+h")
    add("is_post_feriado",       "Calendario / holidays.Peru + holidays.US", "1 si el día anterior a t+h es feriado PE o US",   None, "t+h")
    add("is_pre_eleccion",       "Calendario", "1 si t+h está dentro de los 7 días previos a elecciones presidenciales",    None, "t+h")
    add("is_post_eleccion",      "Calendario", "1 si t+h está dentro de los 7 días posteriores a elecciones presidenciales", None, "t+h")

    # ── Target ────────────────────────────────────────────────────────────────
    add("target", "Datos bancarios", "Flujo neto = D(b, t+h) - R(b, t+h). NaN si no hay datos.", None, "t+h")

    df_dict = pd.DataFrame(registros)
    return df_dict


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
    peru_bday, peru_holidays, fechas_elecciones = build_peru_calendar(
        años=PARAMS["años_calendario"],
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
        fechas_elecciones=fechas_elecciones,
    )

    if not matriz.empty:
        print(
            f"\nMatriz generada: {matriz.shape[0]:,} filas x "
            f"{matriz.shape[1]} columnas"
        )
    else:
        print("\nMatriz vacía. Verifique la configuración de PARAMS y los archivos de datos.")

    # Diccionario de variables
    data_dict = build_data_dictionary(PARAMS)
    print(f"\n{'='*70}")
    print("  DICCIONARIO DE VARIABLES")
    print(f"{'='*70}")
    print(data_dict.to_string(index=False))
    print(f"\nTotal variables: {len(data_dict)}")

    # Exportar diccionario a Excel en 1. Data/Clean/
    ruta_dict = PARAMS.get("ruta_diccionario", r"1. Data/Clean/diccionario_variables.xlsx")
    try:
        os.makedirs(os.path.dirname(ruta_dict), exist_ok=True)
        data_dict.to_excel(ruta_dict, index=False, sheet_name="Diccionario")
        logger.info(f"  Diccionario exportado a: {ruta_dict}")
    except Exception as e:
        logger.warning(f"  No se pudo exportar diccionario: {e}")
