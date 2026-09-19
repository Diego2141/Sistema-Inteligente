# -*- coding: utf-8 -*-
"""
aux_test_lag_calendario.py
==========================
¿Un rezago alineado por CALENDARIO aporta información de ESCALA que el modelo
todavía no está usando?

Motivación
----------
aux_diagnostico_oraculo dejó establecido que la señal de escala del modelo está
SUB-AMPLIFICADA: la anchura predicha varía 2.8x entre el decil 1 y el 10 mientras
el error realizado varía 5.7x (elasticidad ~0.59 en vez de 1). El modelo reacciona
en la dirección correcta pero con la mitad de la amplitud necesaria. Le falta una
variable que expanda el rango dinámico.

Candidata: el flujo en la posición equivalente del mes anterior, de hace dos meses
y de hace tres. La hipótesis es que `sigma_22d` —que el modelo ya tiene— PROMEDIA
sobre todas las posiciones del mes y por eso destruye justamente la variación que
hace falta, mientras que un rezago alineado por calendario compara peras con peras.

Este script NO reentrena nada. Responde si vale la pena tocar step001.

Restricción de disponibilidad
-----------------------------
El rezago k del objetivo t+h cae en t+h-21k días hábiles, observable en t solo si
h <= 21k:

    h <= 21     lags 1, 2, 3
    22-42       lags 2, 3
    43-63       lag 3
    64-75       ninguno

Como hay UN MODELO POR h, cada uno usaría exactamente los lags que le tocan. El
script respeta esa máscara: nunca usa un valor que el modelo no podría ver.

El número que decide
--------------------
NO la correlación simple —todo lo que mide magnitud correlaciona con todo lo que
mide magnitud— sino la CORRELACIÓN PARCIAL controlando por la anchura predicha:
¿aporta algo POR ENCIMA de lo que el modelo ya usa? Y el cara a cara contra
`sigma_22d`, que es el competidor directo.

Se usan rangos (Spearman) porque el error realizado tiene colas muy pesadas: en el
decil 10 el P90 del |error| supera el ancho completo del intervalo.

Salidas
-------
    2. Output/step006_cqr/cqr_<MODO>/
        test_lag_calendario_<BANCO>_<fecha>.png
        test_lag_calendario_<BANCO>_<fecha>.xlsx
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"
DIR_CQR      = BASE_SISTEMA / "2. Output" / "step006_cqr"
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

# Archivo de encaje de aux_encaje_2. Se lee DIRECTO, sin pasar por la matriz, para
# poder medir la proyección de presión de deadline sin reconstruirla primero.
RUTA_ENCAJE  = (BASE_SISTEMA / "2. Output" / "encaje_bbva" /
                "bbva_encaje_features_modelo.xlsx")

_MODO_SUFIJO = "exp_arctan"
BANCO        = "SISTEMA"

DIR_FOLD = DIR_WFCV / f"fold_{_MODO_SUFIJO}"
DIR_OUT  = DIR_CQR  / f"cqr_{_MODO_SUFIJO}"

LAGS_MESES  = [1, 2, 3]

# Benchmark de RECENCIA PURA: magnitud del flujo en los últimos N días hábiles,
# sin alinear por posición del mes. Es el competidor que decide si el valor del
# candidato viene de ALINEAR por calendario o simplemente de "hubo movimiento
# grande hace poco". Si empatan, el feature correcto es éste: más barato de
# construir y disponible para los 74 horizontes en vez de 62.
VENTANAS_RECENCIA = [3, 5, 22]

# Series adicionales tomadas de la MATRIZ (no de R/D) para probarlas con el mismo
# tratamiento de posición del mes. Se descubren del esquema: si una no está, se
# omite en silencio.
#   exceso_abs_lag1        la RESTRICCIÓN que limita cuánto se puede retirar (BBVA)
#   ccovn_sistema_lag1     capacidad física del SISTEMA — el NIVEL, como control
#   var_ccovn_sistema_lag1 su variación diaria
#   residuo_ccovn_lag1     Δsaldo − flujo_neto: la parte de la variación que NO
#                          explica el flujo
#
# El nivel se incluye solo como control. Es un stock grande con tendencia, así que
# compararlo contra el de hace 1-4 meses queda dominado por el crecimiento del
# sistema y no por el ciclo mensual; la variación y el residuo son diferencias, y
# por lo tanto estacionarias.
#
# El residuo es además el único componente de ccovn utilizable sin circularidad:
# el stock cambia por el flujo que se está prediciendo (Δstock = flujo + residuo),
# de modo que proyectar el nivel equivale a predecir el target. El residuo es, por
# construcción, lo que queda una vez descontado el flujo.
SERIES_EXTRA_MATRIZ = ["exceso_abs_lag1", "ccovn_sistema_lag1",
                       "var_ccovn_sistema_lag1", "residuo_ccovn_lag1"]

# Variante de FRECUENCIA: en vez de la magnitud del extremo, en cuántos de los
# últimos 12 meses la posición equivalente superó su umbral. Para modelar colas la
# recurrencia puede ser mejor estadístico que la magnitud. Bonus: con k hasta 12 la
# máscara h <= 21k cubre los 74 horizontes con al menos un rezago.
LAGS_FRECUENCIA = list(range(1, 13))
VENTANA_UMBRAL  = 250   # ruedas para el umbral móvil (point-in-time)

# Controles de CALENDARIO. La parcial contra `ancho` y `h` no basta: varios
# candidatos están estructuralmente ligados a la posición de t+h en el mes.
# presion_deadline es el caso extremo — su denominador, dias_restantes(t+h), es
# calendario puro, así que sin este control podría estar midiendo lo que el modelo
# ya sabe por dias_al_cierre_mes_sin/cos, que es su feature más importante.
# Se leen de la matriz por (fecha_t, h), no como serie diaria: están anclados en
# t+h y por lo tanto varían dentro de una misma fecha de origen.
CONTROLES_CALENDARIO = ["dias_al_cierre_mes_sin", "dias_al_cierre_mes_cos"]
REC_REFERENCIA    = "rec5_flujo"   # el que se usa como control en la parcial

# Competidores: variables de escala que el modelo YA tiene. Se toman las que
# existan en la matriz, en este orden de preferencia para el titular. La matriz
# cambia entre experimentos (la reducida no trae sigma_22d), así que se descubren
# del esquema en vez de fijarlas.
COMPETIDORES_POSIBLES = ["sigma_22d", "sigma_flujo_ratio", "garch_vol",
                         "ma_flujo_20d", "ma_flujo_5d", "sigma_flujo"]
BOOT_B      = 300
BOOT_BLOQUE = 22
SEMILLA     = 20260810
N_BINS      = 10

# Paleta validada (light, surface #fcfcfb): lightness/chroma/CVD/normal-vision PASS
C_CAND  = "#eb6834"   # candidato: rezago de calendario
C_COMP  = "#2a78d6"   # competidor: sigma_22d (lo que el modelo ya tiene)
C_REC   = "#1baf7a"   # benchmark de recencia pura (siempre con etiqueta directa)
C_REF   = "#898781"
C_TINTA = "#0b0b0b"
C_TINTA2 = "#52514e"
C_GRID  = "#e1e0d9"
SURFACE = "#fcfcfb"

EPS = 1e-9
NOMBRE_COMPETIDOR = None   # se resuelve en cargar_preds() según lo hallado


def _titulo(t: str) -> None:
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


# ── Estadística ───────────────────────────────────────────────────────────────
def _rangos(x):
    """Rango normalizado; NaN se mantiene NaN."""
    s = pd.Series(x)
    return (s.rank(method="average") / s.notna().sum()).values


def _matriz_rangos(df, cols):
    """
    Rangos de todas las columnas de una vez, más la máscara de filas completas.

    Se pre-calcula FUERA del bootstrap: rankear 44k filas seis veces por réplica
    domina el costo y vuelve el script inusable con B=300.
    """
    M = np.column_stack([_rangos(df[c]) for c in cols])
    ok = np.isfinite(M).all(axis=1)
    return M, ok


def _parcial_M(M, filas=None):
    """
    Correlación parcial sobre una matriz ya rankeada: col 0 vs col 1,
    controlando por las columnas 2..k. `filas` selecciona un submuestreo.
    """
    A = M if filas is None else M[filas]
    if len(A) < 30:
        return np.nan
    x, y = A[:, 0].copy(), A[:, 1].copy()
    if A.shape[1] > 2:
        Z = np.column_stack([np.ones(len(A)), A[:, 2:]])
        x = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if x.std() < EPS or y.std() < EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def corr_parcial(x, y, Z=None):
    """
    Correlación de Spearman entre x e y controlando por Z (lista de series).

    Sobre rangos y no sobre niveles porque el error realizado tiene colas muy
    pesadas: en el decil 10 el P90 del |error| supera el ancho completo del
    intervalo, y una correlación de Pearson la dominarían un puñado de filas.
    """
    cols = [pd.Series(np.asarray(v)).reset_index(drop=True)
            for v in ([x, y] + list(Z or []))]
    M = np.column_stack([_rangos(c) for c in cols])
    ok = np.isfinite(M).all(axis=1)
    return _parcial_M(M[ok])


def bootstrap_M(df, M, ok, B=BOOT_B, L=BOOT_BLOQUE, semilla=SEMILLA):
    """
    IC 90% por bootstrap de bloques móviles sobre FECHAS DE ORIGEN.

    Las filas no son independientes: 74 horizontes comparten origen y los orígenes
    vecinos comparten datos. Un IC analítico sobre n=44,000 sería varias veces
    demasiado angosto y cualquier correlación parecería significativa.

    Se remuestrean FECHAS en bloques de L días consecutivos; los rangos ya vienen
    calculados sobre la muestra completa.
    """
    rng = np.random.default_rng(semilla)
    d = df.loc[ok] if len(ok) == len(df) else df
    pos = np.flatnonzero(ok)

    fechas = np.sort(d["fecha_t"].unique())
    n_f = len(fechas)
    if n_f < 2 * L:
        return np.nan, np.nan
    cod = pd.Series(np.arange(n_f), index=fechas)
    f_cod = d["fecha_t"].map(cod).values

    orden  = np.argsort(f_cod, kind="stable")
    f_ord  = f_cod[orden]
    inicio = np.searchsorted(f_ord, np.arange(n_f), side="left")
    fin    = np.searchsorted(f_ord, np.arange(n_f), side="right")

    n_bloques = int(np.ceil(n_f / L))
    out = []
    for _ in range(B):
        arr = rng.integers(0, max(1, n_f - L + 1), size=n_bloques)
        fb  = np.concatenate([np.arange(a, min(a + L, n_f)) for a in arr])
        loc = np.concatenate([orden[inicio[f]:fin[f]] for f in fb])
        v = _parcial_M(M[pos], loc)
        if np.isfinite(v):
            out.append(v)
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


# ── 1. Serie diaria de flujo + calendario ─────────────────────────────────────
def _columnas_matriz():
    """Nombres de columna de la matriz sin cargarla (solo el esquema)."""
    try:
        import pyarrow.dataset as pads
        return list(pads.dataset(str(RUTA_MATRIZ)).schema.names)
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(str(RUTA_MATRIZ)).names)
    except Exception:
        return list(pd.read_parquet(RUTA_MATRIZ).columns)


def cargar_serie_diaria():
    """
    Serie diaria de flujo desde la matriz de features.

    Una fila por fecha_t (la matriz tiene 74 filas por fecha, una por horizonte;
    R_t0/D_t0 son constantes dentro de la fecha porque están anclados en t).

    Los competidores se descubren del esquema: la matriz cambia entre
    experimentos y pedir una columna inexistente aborta la lectura entera.
    """
    disponibles = _columnas_matriz()
    comp = [c for c in COMPETIDORES_POSIBLES if c in disponibles]
    faltan = [c for c in ("fecha_t", "R_t0", "D_t0") if c not in disponibles]
    if faltan:
        raise KeyError(f"La matriz no tiene {faltan} — sin ellas no hay serie de flujo")
    if not comp:
        print("[AVISO] la matriz no trae ninguna variable de escala conocida; "
              "el test solo controlará por la anchura predicha y h")
    else:
        print(f"[OK] competidores hallados en la matriz: {comp}")

    extra = [c for c in SERIES_EXTRA_MATRIZ if c in disponibles]
    if extra:
        print(f"[OK] series extra halladas en la matriz: {extra}")
    cols = ["fecha_t", "R_t0", "D_t0"] + comp + extra
    try:
        mat = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", BANCO)],
                              columns=cols)
    except Exception:
        mat = pd.read_parquet(RUTA_MATRIZ, columns=cols + ["banco"])
        mat = mat[mat["banco"] == BANCO]

    mat["fecha_t"] = pd.to_datetime(mat["fecha_t"])
    d = (mat.drop_duplicates("fecha_t")
            .sort_values("fecha_t")
            .set_index("fecha_t"))

    serie = pd.DataFrame(index=d.index)
    serie["flujo"]    = d["D_t0"] - d["R_t0"]    # neto, consistente con el target
    serie["retiro"]   = d["R_t0"]                 # solo R: dirige la cola BAJA
    serie["deposito"] = d["D_t0"]                 # solo D: dirige la cola ALTA
    # Los tres no son redundantes entre sí: el feature es un MÁXIMO de valores
    # absolutos, y max|D-R| no se deduce de max|D| y max|R|.
    for c in comp:
        serie[c] = d[c]
    serie.attrs["competidores"] = comp
    for c in extra:
        serie[c] = d[c]
    serie.attrs["extra"] = extra

    # Recencia pura: mediana de |flujo| en las últimas N ruedas, incluyendo t.
    # Incluir t es correcto y consistente con la máscara de los rezagos: D_t0 y
    # R_t0 están anclados en t y son conocidos ese mismo día.
    for w in VENTANAS_RECENCIA:
        for _c in ("flujo", "retiro", "deposito"):
            serie[f"rec{w}_{_c}"] = serie[_c].abs().rolling(
                w, min_periods=max(2, w // 3)).median()

    # Distancia al cierre de mes, en las DOS métricas. Cuál corresponde es una
    # pregunta abierta: el período de encaje se computa sobre días calendario,
    # pero el modelo lleva dias_al_cierre_mes en hábiles. Se prueban ambas.
    idx = serie.index
    fin_mes = idx + pd.offsets.MonthEnd(0)
    serie["dcm_cal"] = (fin_mes - idx).days

    df_h = pd.DataFrame({"f": idx, "ym": idx.to_period("M")})
    # rank descendente dentro del mes: el último día hábil = 0
    serie["dcm_hab"] = (df_h.groupby("ym")["f"]
                            .rank(ascending=False, method="first").values - 1).astype(int)
    serie["ym"] = idx.to_period("M")

    # Umbral móvil point-in-time para la variante de frecuencia
    for c in ["flujo", "retiro", "deposito"] + extra:
        serie[f"umbral_{c}"] = (serie[c].abs()
                                .rolling(VENTANA_UMBRAL, min_periods=60).median())

    print(f"[OK] serie diaria: {len(serie):,} días | "
          f"{serie.index.min().date()} → {serie.index.max().date()}")
    return serie, comp


def construir_lags(serie, col_flujo, modo_dcm):
    """
    {(año-mes, dcm) -> fecha} para localizar el día equivalente de meses previos.

    Alineando por distancia AL CIERRE, no por posición desde el inicio: los meses
    tienen entre 19 y 23 días hábiles, y anclar al final hace que el último día
    siempre sea 0 — que además es donde se ata la restricción de encaje.
    """
    key = f"dcm_{modo_dcm}"
    idx_map = {}
    for f, ym, dcm in zip(serie.index, serie["ym"], serie[key]):
        idx_map.setdefault((ym, int(dcm)), f)     # primera ocurrencia
    valores = serie[col_flujo]
    return idx_map, valores


def agregar_frecuencia(df, serie, col_serie, etiqueta):
    """
    Fracción de los últimos 12 rezagos de posición cuyo |valor| superó el umbral
    móvil vigente en t. Mide RECURRENCIA, no magnitud: una posición cargada en 8
    de 12 meses es distinta de una cargada en 1 con un valor enorme, y para
    modelar colas la primera puede ser más informativa.

    El umbral es la mediana móvil de 250 ruedas hasta t, así que es point-in-time:
    nunca usa información posterior a la fecha de origen.
    """
    idx_map, valores = construir_lags(serie, col_serie, "hab")
    dcm_th = df["fecha_th"].map(serie["dcm_hab"])
    ym_th  = df["fecha_th"].dt.to_period("M")
    umbral = df["fecha_t"].map(serie[f"umbral_{col_serie}"])

    sup, disp = None, None
    for k in LAGS_FRECUENCIA:
        claves = list(zip(ym_th - k, dcm_th.fillna(-1).astype(int)))
        fechas = pd.to_datetime(pd.Series([idx_map.get(c) for c in claves],
                                          index=df.index))
        vals = fechas.map(valores).abs()
        vals = vals.where(fechas.notna() & (fechas <= df["fecha_t"]))
        ok   = vals.notna() & umbral.notna()
        s_k  = (ok & (vals > umbral)).astype(float).where(ok)
        sup  = s_k if sup is None else sup.add(s_k, fill_value=0)
        disp = ok.astype(float) if disp is None else disp.add(ok.astype(float),
                                                              fill_value=0)
    df[f"esc_{etiqueta}"] = (sup / disp.replace(0, np.nan))
    print(f"     {etiqueta:>18}: frecuencia sobre {len(LAGS_FRECUENCIA)} rezagos | "
          f"cobertura {df[f'esc_{etiqueta}'].notna().mean():.1%}")
    return df, [etiqueta]


def agregar_presion_deadline(df):
    """
    Proyección de la PRESIÓN DE DEADLINE del encaje a la fecha objetivo.

        presion_th = max(ExigibleTotal(t-1) - EncajeAcumMes(t-1), 0)
                     ------------------------------------------------
                              max(dias_restantes(t+h), 1)

    Es el mínimo que el banco tendría que depositar por día a partir de t+h si
    dejara de acumular desde hoy. A diferencia de un rezago histórico, usa el
    estado ACTUAL y solo proyecta el denominador, que es puro calendario — por eso
    no necesita ningún supuesto sobre cómo acumula el banco.

    Dos garantías:

    · Point-in-time. El numerador se toma con shift(1): ExigibleTotalMes_est y
      EncajeAcumMes en t no se conocen hasta t+1. El denominador es calendario
      puro, así que no requiere dato alguno.

    · Validez. El exceso se REINICIA al cierre del período de encaje, de modo que
      la proyección solo vale si t+h cae en el mismo mes calendario que t. Fuera
      de eso queda NaN: cubre ~14% de las filas, concentradas en h<=10, que es
      donde el allocation lo necesita (91% de los orígenes en h=2, 53% en h=10).
    """
    if not RUTA_ENCAJE.exists():
        print(f"[AVISO] sin {RUTA_ENCAJE.name} — se omite presion_deadline_th")
        return df, []
    try:
        enc = pd.read_excel(RUTA_ENCAJE)
    except Exception as e:
        print(f"[AVISO] no se pudo leer el archivo de encaje ({e})")
        return df, []

    faltan = [c for c in ("fecha", "ExigibleTotalMes_est", "EncajeAcumMes")
              if c not in enc.columns]
    if faltan:
        print(f"[AVISO] al archivo de encaje le faltan {faltan} — se omite")
        return df, []

    enc["fecha"] = pd.to_datetime(enc["fecha"])
    enc = enc.dropna(subset=["fecha"]).sort_values("fecha").set_index("fecha")

    # shift(1): el estado de t recién se conoce en t+1
    deficit = (enc["ExigibleTotalMes_est"] - enc["EncajeAcumMes"]).clip(lower=0).shift(1)
    # Días calendario → ffill a días hábiles, igual que hace step001
    deficit = deficit.reindex(
        pd.date_range(enc.index.min(), enc.index.max(), freq="D")).ffill()

    th = pd.DatetimeIndex(df["fecha_th"])
    ft = pd.DatetimeIndex(df["fecha_t"])
    # dias_restantes(t+h): calendario puro, no necesita dato
    fin_mes = th + pd.offsets.MonthEnd(0)
    dias_rest_th = (fin_mes - th).days

    num = pd.Series(ft, index=df.index).map(deficit)
    val = num / np.maximum(dias_rest_th, 1)
    # Solo válido dentro del mismo período de encaje
    mismo = (th.to_period("M") == ft.to_period("M"))
    df["esc_presion_deadline"] = val.where(mismo)

    cob = df["esc_presion_deadline"].notna().mean()
    print(f"     {'presion_deadline':>18}: proyección de deadline | "
          f"cobertura {cob:.1%} (solo dentro del período de encaje)")
    return df, ["presion_deadline"]


# ── 2. Muestra: predicciones de TEST ──────────────────────────────────────────
def cargar_preds(serie, comp):
    paths = sorted(DIR_FOLD.glob(f"preds_base_{BANCO}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Sin preds_base en {DIR_FOLD}")
    df = pd.read_parquet(paths[-1])
    print(f"[OK] {paths[-1].name}: {len(df):,} filas")

    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.dropna(subset=["target", "q05", "q50", "q95"]).copy()

    # fecha_th: si no viene, se deriva del calendario hábil de la propia serie
    if "fecha_th" in df.columns and df["fecha_th"].notna().any():
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])
    else:
        dias = np.array(serie.index)
        pos  = pd.Series(np.arange(len(dias)), index=serie.index)
        p_t  = df["fecha_t"].map(pos)
        p_th = p_t + df["h"]
        ok   = p_t.notna() & (p_th < len(dias))
        df["fecha_th"] = pd.NaT
        df.loc[ok, "fecha_th"] = dias[p_th[ok].astype(int).values]
        print(f"     fecha_th derivada del calendario ({int(ok.sum()):,} filas)")

    df = df.dropna(subset=["fecha_th"]).copy()

    # Controles de calendario, por (fecha_t, h) — anclados en t+h
    global CAL_OK
    CAL_OK = []
    try:
        _cal = pd.read_parquet(RUTA_MATRIZ,
                               columns=["fecha_t", "h"] + CONTROLES_CALENDARIO)
        _cal["fecha_t"] = pd.to_datetime(_cal["fecha_t"])
        _cal = _cal.drop_duplicates(subset=["fecha_t", "h"])
        df = df.merge(_cal, on=["fecha_t", "h"], how="left")
        CAL_OK = [c for c in CONTROLES_CALENDARIO
                  if c in df.columns and df[c].notna().any()]
        print(f"[OK] controles de calendario: {CAL_OK}")
    except Exception as e:
        print(f"[AVISO] sin controles de calendario ({e}) — la parcial no podrá "
              f"distinguir señal propia de posición del mes")

    # Variables a explicar y control
    df["r"]     = (df["target"] - df["q50"]).abs()          # escala NO capturada
    df["r_lo"]  = (df["q50"] - df["target"]).clip(lower=0)  # solo sorpresas abajo
    df["ancho"] = df["q95"] - df["q05"]                     # escala SÍ predicha

    # Competidor: la mejor variable de escala que el modelo YA tiene, entre las
    # halladas en el esquema de la matriz. Se guarda con el alias interno
    # "sigma_22d" para no reescribir el resto del script — el nombre real se
    # imprime una vez para que quede claro qué se está comparando.
    global NOMBRE_COMPETIDOR
    if comp:
        NOMBRE_COMPETIDOR = comp[0]
        df["sigma_22d"] = df["fecha_t"].map(serie[NOMBRE_COMPETIDOR])
        print(f"[OK] competidor usado: {NOMBRE_COMPETIDOR} "
              f"(alias interno 'sigma_22d' en el resto del reporte)")
    else:
        NOMBRE_COMPETIDOR = None
        df["sigma_22d"] = np.nan
        print("[AVISO] sin competidor disponible en la matriz — las columnas "
              "'sigma_22d' del reporte quedan vacías (comparación omitida)")
    return df


def agregar_candidato(df, serie, col_flujo, modo_dcm, etiqueta,
                      stats=("med", "max", "mea", "med3", "max3", "mea3"),
                      ancla="th"):
    """
    Añade pos_lag_1/2/3 y esc_<etiqueta> respetando la máscara de disponibilidad.

    Un rezago solo se usa si su fecha es <= fecha_t. Nunca se le da al modelo un
    valor que no podría observar; sin esa máscara el test daría un resultado
    espectacular y falso.
    """
    key = f"dcm_{modo_dcm}"
    idx_map, valores = construir_lags(serie, col_flujo, modo_dcm)

    # ancla="th": la posición la define el DÍA QUE SE PREDICE (t+h). Es lo que
    #   hace la matriz con todas sus variables de calendario, y es lo que permite
    #   que el árbol cruce "dónde cae t+h" con "qué tan grande es el flujo ahí".
    # ancla="t" : la posición la define el ORIGEN. El feature queda idéntico para
    #   los 74 horizontes del mismo origen, así que no informa sobre cuál día se
    #   predice — solo sirve como indicador de régimen. Se mide para no dar por
    #   sentada la elección.
    _col_ancla = "fecha_th" if ancla == "th" else "fecha_t"
    dcm_th = df[_col_ancla].map(serie[key])
    ym_th  = df[_col_ancla].dt.to_period("M")

    cols_abs, n_disp = [], []
    for k in LAGS_MESES:
        claves = list(zip(ym_th - k, dcm_th.fillna(-1).astype(int)))
        fechas = pd.Series([idx_map.get(c) for c in claves], index=df.index)
        fechas = pd.to_datetime(fechas)

        vals = fechas.map(valores)
        # Máscara de disponibilidad point-in-time
        vals = vals.where(fechas.notna() & (fechas <= df["fecha_t"]))

        df[f"lag{k}_{etiqueta}"] = vals
        cols_abs.append(vals.abs())
        n_disp.append(int(vals.notna().sum()))

    A = pd.concat(cols_abs, axis=1)
    n_lags = A.notna().sum(axis=1)
    df[f"n_lags_{etiqueta}"] = n_lags

    # Tres estadísticos sobre los |rezagos| disponibles.
    #
    # MEDIANA: su esperanza NO depende de cuántos rezagos haya. Importa porque el
    #   conteo varía con h (3, 2 o 1) y con el largo del mes.
    # MÁXIMO : mejor alineado con predecir COLAS, pero E[máx de 3] > E[máx de 1],
    #   así que crece con el conteo — que es función de h. Ese sesgo se mide
    #   aparte con la variante "solo filas con los 3 rezagos".
    # MEDIA  : intermedio, se incluye como referencia.
    etiquetas = []
    for suf, serie_est in (("med", A.median(axis=1, skipna=True)),
                           ("max", A.max(axis=1, skipna=True)),
                           ("mea", A.mean(axis=1, skipna=True))):
        if suf not in stats:
            continue
        df[f"esc_{etiqueta}_{suf}"] = serie_est
        etiquetas.append(f"{etiqueta}_{suf}")

    # Los tres estadísticos restringidos a filas con TODOS los rezagos. Es la
    # única comparación limpia entre ellos: sobre la misma submuestra y sin el
    # sesgo de que E[máx de 3] > E[máx de 1]. Cobertura baja a propósito.
    completo = n_lags == len(LAGS_MESES)
    for suf, est in (("med3", A.median(axis=1, skipna=True)),
                     ("max3", A.max(axis=1, skipna=True)),
                     ("mea3", A.mean(axis=1, skipna=True))):
        if suf not in stats:
            continue
        df[f"esc_{etiqueta}_{suf}"] = est.where(completo)
        etiquetas.append(f"{etiqueta}_{suf}")

    cob = df[f"esc_{etiqueta}_{etiquetas[0].split('_')[-1]}"].notna().mean()
    print(f"     {etiqueta:>18}: lags disponibles {n_disp} | "
          f"cobertura {cob:.1%} | estadísticos {[e.split('_')[-1] for e in etiquetas]}")
    return df, etiquetas


# ── 3. Test ───────────────────────────────────────────────────────────────────
def main() -> None:
    serie, comp = cargar_serie_diaria()
    df = cargar_preds(serie, comp)

    _titulo("Construcción de candidatos (con máscara point-in-time)")
    variantes, cal_vars, rec_vars = [], [], []
    for col_flujo in ("flujo", "retiro", "deposito"):
        # 'cal' ya perdió decisivamente contra 'hab'; se mantiene solo como
        # referencia para las dos series originales, no se extiende a depósitos.
        for modo in (("hab", "cal") if col_flujo != "deposito" else ("hab",)):
            et = f"{col_flujo}_{modo}"
            # Los tres estadísticos solo para la alineación hábil, que ya ganó
            # decisivamente; para 'cal' basta la mediana como referencia.
            df, ets = agregar_candidato(
                df, serie, col_flujo, modo, et,
                stats=("med",) if modo == "cal"
                      else ("med", "max", "mea", "med3", "max3", "mea3"))
            variantes.extend(ets); cal_vars.extend(ets)

    # Misma construcción pero anclada en el ORIGEN, para verificar que anclar en
    # el día predicho es lo correcto y no una suposición.
    for col_flujo in ("flujo", "retiro", "deposito"):
        et = f"{col_flujo}_anclaT"
        df, ets = agregar_candidato(df, serie, col_flujo, "hab", et,
                                    stats=("med", "max"), ancla="t")
        variantes.extend(ets); cal_vars.extend(ets)

    # Series de la matriz (restricción de encaje y capacidad del sistema) con el
    # mismo tratamiento de posición del mes. Solo med y max: el objetivo es saber
    # si la FAMILIA aporta, no afinar el estadístico todavía.
    for _c in serie.attrs.get("extra", []):
        df, ets = agregar_candidato(df, serie, _c, "hab", _c,
                                    stats=("med", "max"))
        variantes.extend(ets); cal_vars.extend(ets)

    # Variante de recurrencia sobre las tres series. Depósitos incluido: la
    # permutación del modelo puso esc_deposito_pos entre los primeros, así que
    # omitirlo aquí no tendría justificación.
    for _c in ("retiro", "deposito", "flujo"):
        df, ets = agregar_frecuencia(df, serie, _c, f"frec_{_c}")
        variantes.extend(ets); cal_vars.extend(ets)

    # Proyección de presión de deadline (no es un rezago: usa el estado actual)
    df, ets = agregar_presion_deadline(df)
    variantes.extend(ets); cal_vars.extend(ets)

    # Recencia pura: se mapea por fecha_t, así que está disponible siempre y para
    # todos los horizontes. Comparte el prefijo esc_ para pasar por las mismas
    # secciones sin código aparte.
    for w in VENTANAS_RECENCIA:
        for serie_col in ("flujo", "retiro", "deposito"):
            et = f"rec{w}_{serie_col}"
            df[f"esc_{et}"] = df["fecha_t"].map(serie[et])
            variantes.append(et); rec_vars.append(et)
            print(f"     {et:>18}: recencia pura | "
                  f"cobertura {df[f'esc_{et}'].notna().mean():.1%}")

    _titulo("A · ¿Aporta escala POR ENCIMA de lo que el modelo ya usa?")
    print("  Spearman contra r = |target − q50|, controlando por anchura Y por h\n")
    _et_comp = NOMBRE_COMPETIDOR or "sin competidor"
    _col_rec = f"esc_{REC_REFERENCIA}"
    print(f"  {'candidato':>18} {'simple':>9} {'| ancho':>10} "
          f"{'IC 90%':>18} {'| +'+_et_comp[:9]:>13} {'| +recen':>9} "
          f"{'| +CALEND':>10}")
    print("  " + "-" * 76)

    filas = []
    for et in variantes + ["sigma_22d"]:
        col = et if et == "sigma_22d" else f"esc_{et}"
        sub = df.dropna(subset=[col, "r", "ancho"])
        if len(sub) < 100:
            continue
        c_simple = corr_parcial(sub[col], sub["r"])
        M, ok    = _matriz_rangos(sub, [col, "r", "ancho", "h"])
        c_anch   = _parcial_M(M[ok])
        lo, hi   = bootstrap_M(sub, M, ok)
        sub2 = sub.dropna(subset=["sigma_22d"])
        c_both = (np.nan if et == "sigma_22d" or len(sub2) < 100 else
                  corr_parcial(sub2[col], sub2["r"],
                               [sub2["ancho"], sub2["h"], sub2["sigma_22d"]]))
        # ¿Sobrevive a controlar por la RECENCIA PURA? Es la pregunta que separa
        # "alinear por calendario aporta" de "solo detecta movimiento reciente".
        if et in rec_vars or et == "sigma_22d":
            c_rec = np.nan
        else:
            sub3 = sub.dropna(subset=[_col_rec])
            c_rec = (np.nan if len(sub3) < 100 else
                     corr_parcial(sub3[col], sub3["r"],
                                  [sub3["ancho"], sub3["h"], sub3[_col_rec]]))
        # ¿Sobrevive a controlar por la POSICIÓN EN EL MES? Es lo que separa
        # "aporta señal propia" de "está midiendo el calendario que el modelo ya
        # tiene". Decisivo para presion_deadline, cuyo denominador es calendario
        # puro, y relevante para todo lo que se ancla en la posición de t+h.
        if CAL_OK:
            sub4 = sub.dropna(subset=CAL_OK)
            c_cal = (np.nan if len(sub4) < 100 else
                     corr_parcial(sub4[col], sub4["r"],
                                  [sub4["ancho"], sub4["h"]]
                                  + [sub4[c] for c in CAL_OK]))
        else:
            c_cal = np.nan

        if et == "sigma_22d":
            marca = f"  <- competidor ({NOMBRE_COMPETIDOR})"
        elif et in rec_vars:
            marca = "  <- recencia pura"
        else:
            marca = ""
        print(f"  {et:>18} {c_simple:>+9.3f} {c_anch:>+10.3f} "
              f"[{lo:>+6.3f},{hi:>+6.3f}] "
              f"{'—' if not np.isfinite(c_both) else f'{c_both:+.3f}':>13} "
              f"{'—' if not np.isfinite(c_rec) else f'{c_rec:+.3f}':>9} "
              f"{'—' if not np.isfinite(c_cal) else f'{c_cal:+.3f}':>10}{marca}")
        filas.append({"candidato": et, "n": len(sub),
                      "cobertura": len(sub) / len(df),
                      "corr_simple": c_simple,
                      "parcial_ancho": c_anch, "ic_lo": lo, "ic_hi": hi,
                      "parcial_ancho_sigma": c_both, "parcial_ancho_recencia": c_rec,
                      "parcial_ancho_calendario": c_cal,
                      "tipo": ("competidor" if et == "sigma_22d"
                               else "recencia" if et in rec_vars else "calendario")})
    tab = pd.DataFrame(filas)

    print("\n  'simple' sube por construcción: todo lo que mide magnitud")
    print("  correlaciona con todo lo que mide magnitud. El número que decide es")
    print("  '| ancho': lo que aporta POR ENCIMA de la escala que el modelo ya predice.")
    print(f"  '| +{_et_comp}' responde si le gana a promediar sobre el mes.")
    print(f"  '| +recen' responde si le gana a mirar solo lo reciente ({REC_REFERENCIA}).")
    print("  '| +CALEND' es el más exigente: controla además por la posición de t+h")
    print("  en el mes. Si un candidato se derrumba ahí, no aportaba señal propia —")
    print("  estaba midiendo el calendario, que el modelo ya tiene en sin/cos.")

    # ── A2. Cara a cara en SUBMUESTREO COMÚN ──────────────────────────────────
    _titulo("A2 · Calendario vs recencia pura, sobre las MISMAS filas")
    print("  El candidato de calendario cubre ~78% de las filas y la recencia el")
    print("  100%: compararlos sobre muestras distintas sería tramposo. Acá se")
    print("  restringe a las filas donde AMBOS existen.\n")

    mejor_cal = max(cal_vars,
                    key=lambda e: abs(tab.loc[tab["candidato"] == e,
                                              "parcial_ancho"].iloc[0])
                    if (tab["candidato"] == e).any() else 0)
    col_cal = f"esc_{mejor_cal}"
    print(f"  {'variable':>18} {'tipo':>11} {'| ancho,h':>11} {'IC 90%':>18}")
    print("  " + "-" * 62)
    filas_cc, comunes = [], df.dropna(subset=[col_cal, "r", "ancho"])
    for et in [mejor_cal] + rec_vars:
        col = f"esc_{et}"
        sub_c = comunes.dropna(subset=[col])
        if len(sub_c) < 100:
            continue
        M, ok = _matriz_rangos(sub_c, [col, "r", "ancho", "h"])
        c = _parcial_M(M[ok])
        lo, hi = bootstrap_M(sub_c, M, ok)
        tipo = "CALENDARIO" if et == mejor_cal else "recencia"
        print(f"  {et:>18} {tipo:>11} {c:>+11.3f} [{lo:>+6.3f},{hi:>+6.3f}]")
        filas_cc.append({"variable": et, "tipo": tipo, "n": len(sub_c),
                         "parcial": c, "ic_lo": lo, "ic_hi": hi})
    tab_cc = pd.DataFrame(filas_cc)
    print(f"\n  n común = {len(comunes):,} filas")

    # Veredicto del cara a cara: se compara el punto del calendario contra el
    # TECHO del IC de la mejor recencia. Si aun así gana, la ventaja no se
    # explica por ruido de muestreo.
    if len(tab_cc) >= 2:
        f_cal = tab_cc[tab_cc["tipo"] == "CALENDARIO"].iloc[0]
        f_rec = tab_cc[tab_cc["tipo"] == "recencia"].sort_values(
            "parcial", ascending=False).iloc[0]
        print(f"\n  calendario {f_cal['parcial']:+.3f}  vs  "
              f"mejor recencia ({f_rec['variable']}) {f_rec['parcial']:+.3f}")
        if f_cal["parcial"] > f_rec["ic_hi"]:
            print("  -> El calendario gana incluso contra el TECHO del IC de la")
            print("     recencia. La alineación aporta valor propio.")
        elif f_cal["ic_hi"] < f_rec["parcial"]:
            print("  -> La RECENCIA gana. Alinear por calendario no aporta: el")
            print("     feature correcto es la magnitud reciente, más barata y")
            print("     disponible en los 74 horizontes.")
        else:
            print("  -> EMPATE dentro del ruido. Con ICs superpuestos no hay")
            print("     evidencia para pagar el costo de la alineación; conviene")
            print("     la recencia por parsimonia y cobertura.")

    # ── A3. ¿Anclar en t+h o en t? ────────────────────────────────────────────
    _titulo("A3 · Anclaje: ¿la posición la define el día PREDICHO o el ORIGEN?")
    print("  Toda variable de calendario de la matriz está anclada en t+h, porque")
    print("  el target es el flujo DE ESE DÍA. La alternativa es anclar en el")
    print("  origen, con la ventaja de estar siempre disponible. Se compara sobre")
    print("  las MISMAS filas y con el MISMO estadístico.\n")
    print(f"  {'serie':>8} {'estad.':>7} | {'ancla t+h':>10} {'IC 90%':>18} | "
          f"{'ancla t':>9} {'IC 90%':>18}")
    print("  " + "-" * 82)
    filas_a3 = []
    for serie_col in ("flujo", "retiro", "deposito"):
        for est in ("med", "max"):
            c_th, c_t = f"esc_{serie_col}_hab_{est}", f"esc_{serie_col}_anclaT_{est}"
            if c_th not in df.columns or c_t not in df.columns:
                continue
            com = df.dropna(subset=[c_th, c_t, "r", "ancho"])
            if len(com) < 100:
                continue
            vals = {}
            for nom, col in (("th", c_th), ("t", c_t)):
                M, ok = _matriz_rangos(com, [col, "r", "ancho", "h"])
                vals[nom] = (_parcial_M(M[ok]),) + bootstrap_M(com, M, ok)
            print(f"  {serie_col:>8} {est:>7} | {vals['th'][0]:>+10.3f} "
                  f"[{vals['th'][1]:>+6.3f},{vals['th'][2]:>+6.3f}] | "
                  f"{vals['t'][0]:>+9.3f} "
                  f"[{vals['t'][1]:>+6.3f},{vals['t'][2]:>+6.3f}]")
            filas_a3.append({"serie": serie_col, "estadistico": est, "n": len(com),
                             "parcial_ancla_th": vals["th"][0],
                             "ic_lo_th": vals["th"][1], "ic_hi_th": vals["th"][2],
                             "parcial_ancla_t": vals["t"][0],
                             "ic_lo_t": vals["t"][1], "ic_hi_t": vals["t"][2]})
    tab_a3 = pd.DataFrame(filas_a3)
    if not tab_a3.empty:
        gana_th = (tab_a3["parcial_ancla_th"].abs()
                   > tab_a3["parcial_ancla_t"].abs()).sum()
        print(f"\n  ancla t+h gana en {gana_th}/{len(tab_a3)} comparaciones")
        sep = (tab_a3["ic_lo_th"] > tab_a3["ic_hi_t"]).sum()
        print(f"  con ICs que NO se solapan: {sep}/{len(tab_a3)}")
        if sep >= len(tab_a3) - 1:
            print("  -> Anclar en el día PREDICHO es lo correcto, y con margen.")
        elif gana_th >= len(tab_a3) - 1:
            print("  -> t+h gana consistentemente, pero los ICs se solapan: la")
            print("     evidencia apoya la elección sin ser concluyente.")
        else:
            print("  -> Los dos anclajes rinden parecido. Anclar en t tiene la")
            print("     ventaja de estar disponible en los 74 horizontes, así que")
            print("     habría que reconsiderar la elección.")

    # ── B. Solo la cola baja (lo operativo) ───────────────────────────────────
    _titulo("B · Ídem contra r_lo = max(q50 − target, 0)  — solo retiros")
    print(f"  {'candidato':>18} {'| ancho':>10} {'IC 90%':>18}")
    print("  " + "-" * 50)
    for et in variantes + ["sigma_22d"]:
        col = et if et == "sigma_22d" else f"esc_{et}"
        sub = df.dropna(subset=[col, "r_lo", "ancho"])
        if len(sub) < 100:
            continue
        M, ok  = _matriz_rangos(sub, [col, "r_lo", "ancho", "h"])
        c      = _parcial_M(M[ok])
        lo, hi = bootstrap_M(sub, M, ok)
        tab.loc[tab["candidato"] == et, "parcial_lo"] = c
        print(f"  {et:>18} {c:>+10.3f} [{lo:>+6.3f},{hi:>+6.3f}]")

    # ── B2. Contra el decil superior del error: ¿mediana o máximo? ────────────
    _titulo("B2 · ¿Qué estadístico predice los EXTREMOS?  (mediana vs máximo)")
    print("  Las secciones A y B correlacionan contra |error|, que es una medida")
    print("  de CENTRO — eso favorece a la mediana por construcción del test. Si")
    print("  lo que importa son las colas, el objetivo correcto es un indicador de")
    print("  error extremo. Se usa el decil superior DENTRO de cada h, porque el")
    print("  error escala con el horizonte y un P90 global solo elegiría h grandes.\n")

    df["r_ext"] = (df.groupby("h")["r"].rank(pct=True) > 0.90).astype(float)
    print(f"  {'candidato':>22} {'vs |error|':>11} {'vs decil sup.':>14} {'IC 90%':>18}")
    print("  " + "-" * 70)
    filas_ext = []
    for et in variantes + ["sigma_22d"]:
        col = et if et == "sigma_22d" else f"esc_{et}"
        sub_e = df.dropna(subset=[col, "r_ext", "ancho"])
        if len(sub_e) < 100:
            continue
        M, ok = _matriz_rangos(sub_e, [col, "r_ext", "ancho", "h"])
        c_ext = _parcial_M(M[ok])
        lo, hi = bootstrap_M(sub_e, M, ok)
        prev = tab.loc[tab["candidato"] == et, "parcial_ancho"]
        c_r = float(prev.iloc[0]) if len(prev) else np.nan
        print(f"  {et:>22} {c_r:>+11.3f} {c_ext:>+14.3f} [{lo:>+6.3f},{hi:>+6.3f}]")
        filas_ext.append({"candidato": et, "parcial_r": c_r,
                          "parcial_extremo": c_ext, "ic_lo": lo, "ic_hi": hi})
    tab_ext = pd.DataFrame(filas_ext)

    # Comparación directa entre estadísticos de la MISMA serie y alineación
    def _tabla_estadisticos(tabla, col_val, titulo):
        print(f"\n  {titulo}")
        print(f"  {'familia':>13} | {'med':>7} {'max':>7} {'mea':>7} | "
              f"{'med3':>7} {'max3':>7} {'mea3':>7} | {'gana(3)':>8}")
        print("  " + "-" * 76)
        for fam in ("flujo_hab", "retiro_hab", "deposito_hab"):
            v = {}
            for suf in ("med", "max", "mea", "med3", "max3", "mea3"):
                f_ = tabla[tabla["candidato"] == f"{fam}_{suf}"]
                v[suf] = float(f_[col_val].iloc[0]) if len(f_) else np.nan
            tri = {k: v[k] for k in ("med3", "max3", "mea3") if np.isfinite(v[k])}
            gana = max(tri, key=tri.get) if tri else "—"
            print(f"  {fam:>13} | {v['med']:>+7.3f} {v['max']:>+7.3f} {v['mea']:>+7.3f} | "
                  f"{v['med3']:>+7.3f} {v['max3']:>+7.3f} {v['mea3']:>+7.3f} | {gana:>8}")

    _tabla_estadisticos(tab_ext, "parcial_extremo", "Contra el DECIL SUPERIOR del error:")
    _tabla_estadisticos(tab, "parcial_ancho", "Contra |error| (referencia, sección A):")

    print("\n  Las tres primeras columnas usan TODA la muestra pero mezclan filas")
    print("  con 3, 2 y 1 rezago — y con 1 solo rezago med = max = mea, así que")
    print("  ahí los tres son idénticos por construcción y el contraste se diluye.")
    print("  Las columnas *3 restringen a filas con los 3 rezagos: misma submuestra,")
    print("  sin el sesgo de que E[máx de 3] > E[máx de 1]. Esa es la comparación")
    print("  limpia entre estadísticos, y por eso 'gana(3)' se decide sobre ellas.")

    # ── C. Por tramo de horizonte ─────────────────────────────────────────────
    _titulo("C · Por tramo de horizonte (la disponibilidad cambia con h)")
    # El "mejor" se elige SOLO entre candidatos de cobertura razonable. Las
    # variantes *3 están restringidas a filas con los 3 rezagos (h<=21), así que
    # su parcial no es comparable con la de un candidato de cobertura completa:
    # elegirlas por magnitud bruta distorsionaría A2, C y D, que se calculan
    # sobre la submuestra del ganador.
    COBERTURA_MIN = 0.50
    _elegibles = tab[tab["cobertura"] >= COBERTURA_MIN]
    if _elegibles.empty:
        _elegibles = tab
    mejor = _elegibles.loc[_elegibles["parcial_ancho"].abs().idxmax(), "candidato"]
    col_m = mejor if mejor == "sigma_22d" else f"esc_{mejor}"
    _descartados = tab.loc[tab["cobertura"] < COBERTURA_MIN, "candidato"].tolist()
    if _descartados:
        print(f"  (excluidos por cobertura <{COBERTURA_MIN:.0%}: "
              f"{', '.join(_descartados)})")
    print(f"  Mejor candidato de la sección A: {mejor}\n")
    df["h_bin"] = pd.cut(df["h"], [1, 21, 42, 63, 75],
                         labels=["2-21 (3 lags)", "22-42 (2 lags)",
                                 "43-63 (1 lag)", "64-75 (ninguno)"])
    print(f"  {'tramo':>18} {'n':>8} {'cob.':>7} {'| ancho':>10} "
          f"{(NOMBRE_COMPETIDOR or 'competidor')[:11]:>11}")
    print("  " + "-" * 60)
    filas_h = []
    for tramo, g in df.groupby("h_bin", observed=True):
        s = g.dropna(subset=[col_m, "r", "ancho"])
        cob = g[col_m].notna().mean()
        c  = corr_parcial(s[col_m], s["r"], [s["ancho"]]) if len(s) >= 100 else np.nan
        s2 = g.dropna(subset=["sigma_22d", "r", "ancho"])
        cs = corr_parcial(s2["sigma_22d"], s2["r"], [s2["ancho"]]) if len(s2) >= 100 else np.nan
        print(f"  {str(tramo):>18} {len(s):>8,} {cob:>7.0%} "
              f"{c:>+10.3f} {cs:>+11.3f}")
        filas_h.append({"tramo": str(tramo), "n": len(s), "cobertura": cob,
                        "parcial_candidato": c, "parcial_sigma": cs})
    tab_h = pd.DataFrame(filas_h)

    # ── D. ¿Expande el rango dinámico? ────────────────────────────────────────
    _titulo("D · ¿Expandiría el rango dinámico de la anchura?")
    print("  El diagnóstico del oráculo midió elasticidad ~0.59 (el modelo mueve")
    print("  la anchura 2.8x cuando el error se mueve 5.7x). Acá se mide cuánto")
    print("  ordena el candidato el error DENTRO de cada decil de anchura —")
    print("  o sea, la variación que el modelo se está perdiendo.\n")

    sub = df.dropna(subset=[col_m, "r", "ancho"]).copy()
    sub["dec_ancho"] = pd.qcut(sub["ancho"], N_BINS, labels=False, duplicates="drop")
    sub["dec_cand"] = (sub.groupby("dec_ancho")[col_m]
                          .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                                       labels=False,
                                                       duplicates="drop")))
    sub = sub.dropna(subset=["dec_cand"])
    piv = (sub.groupby("dec_cand")["r"].median() / 1e6) if len(sub) else pd.Series(dtype=float)
    if piv.empty:
        n_anchuras = df["ancho"].nunique()
        print(f"  [AVISO] sin quintiles utilizables — se omite la sección "
              f"(anchura predicha con {n_anchuras} valor(es) distinto(s); "
              f"qcut necesita variación real en 'ancho')")
    print(f"  {'quintil del candidato (intra decil de anchura)':>48} {'mediana |error| MM':>20}")
    print("  " + "-" * 70)
    for q, v in piv.items():
        print(f"  {'Q'+str(int(q)+1):>48} {v:>20,.0f}")
    if len(piv) >= 2 and piv.iloc[0] > 0:
        ratio = piv.iloc[-1] / piv.iloc[0]
        print(f"\n  Rango dinámico intra-decil: {ratio:.2f}x")
        print("  >1.3x significa que el candidato separa filas de escala distinta")
        print("  que el modelo hoy trata igual — exactamente lo que falta.")

    # ── Veredicto ─────────────────────────────────────────────────────────────
    _titulo("Veredicto")
    fila_m = tab.loc[tab["candidato"] == mejor].iloc[0]
    fila_s = tab.loc[tab["candidato"] == "sigma_22d"]
    p_sig  = float(fila_s["parcial_ancho"].iloc[0]) if len(fila_s) else np.nan
    # Umbral sobre el PISO del IC, no sobre el punto. 0.05 y no 0.02 porque en el
    # mundo sintetico SIN estructura mensual apareció un espurio de +0.062 antes
    # de controlar por h; el margen protege de lo que quede sin controlar.
    UMBRAL_SIGNIF = 0.05
    signif = np.isfinite(fila_m["ic_lo"]) and (fila_m["ic_lo"] > UMBRAL_SIGNIF)
    _cal_m = fila_m.get("parcial_ancho_calendario", np.nan)
    if np.isfinite(_cal_m) and abs(_cal_m) < UMBRAL_SIGNIF:
        signif = False

    print(f"  Mejor candidato          : {mejor}")
    print(f"  Parcial | ancho          : {fila_m['parcial_ancho']:+.3f} "
          f"[{fila_m['ic_lo']:+.3f}, {fila_m['ic_hi']:+.3f}]")
    print(f"  Competidor ({NOMBRE_COMPETIDOR or 'ninguno'})"
          f"{' ' * max(0, 12 - len(NOMBRE_COMPETIDOR or 'ninguno'))}: {p_sig:+.3f}")
    print(f"  Parcial | ancho + sigma  : {fila_m['parcial_ancho_sigma']:+.3f}")
    if "parcial_ancho_recencia" in fila_m.index and np.isfinite(fila_m["parcial_ancho_recencia"]):
        print(f"  Parcial | ancho + recencia: {fila_m['parcial_ancho_recencia']:+.3f}"
              "   <- si cae a ~0, el valor era recencia, no calendario")
    print()
    if not signif:
        print(f"  (criterio: piso del IC 90% > {UMBRAL_SIGNIF:+.2f})\n")
        print("  -> NO CONSTRUIR. El rezago de calendario no aporta escala por encima")
        print("     de lo que el modelo ya usa. Nos ahorramos tocar step001 y una")
        print("     corrida completa de revalidación.")
    elif np.isfinite(fila_m["parcial_ancho_sigma"]) and abs(fila_m["parcial_ancho_sigma"]) < UMBRAL_SIGNIF:
        print(f"  -> REDUNDANTE. Aporta sobre la anchura, pero NO sobre {NOMBRE_COMPETIDOR}:")
        print("     la información ya está en el modelo y el problema no es que")
        print("     falte la variable sino cómo la usa. Alinear por calendario no")
        print("     le gana a promediar.")
    else:
        print("  -> CONSTRUIR. Aporta información de escala incremental sobre la")
        print(f"     anchura predicha Y sobre {NOMBRE_COMPETIDOR}. Alinear por calendario sí")
        print("     preserva variación que promediar destruye.")
        # Parsear por partes: el nombre es <serie>_<alineación>_<estadístico>
        _p = mejor.split("_")
        print(f"     Serie ganadora     : "
              f"{'flujo neto' if _p[0] == 'flujo' else 'retiro'}")
        print(f"     Alineación ganadora: "
              f"{'días hábiles' if len(_p) > 1 and _p[1] == 'hab' else 'días calendario'}")
        print(f"     Estadístico ganador: {_p[2] if len(_p) > 2 else 'mediana'}")

    # ── Salidas ───────────────────────────────────────────────────────────────
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    fecha = date.today().strftime("%Y%m%d")
    ruta_xl = DIR_OUT / f"test_lag_calendario_{BANCO}_{fecha}.xlsx"
    try:
        with pd.ExcelWriter(ruta_xl, engine="openpyxl") as xl:
            tab.to_excel(xl, sheet_name="candidatos", index=False)
            tab_h.to_excel(xl, sheet_name="por_horizonte", index=False)
            tab_ext.to_excel(xl, sheet_name="extremos", index=False)
            if not tab_a3.empty:
                tab_a3.to_excel(xl, sheet_name="anclaje", index=False)
            tab_cc.to_excel(xl, sheet_name="cara_a_cara", index=False)
            piv.reset_index().to_excel(xl, sheet_name="rango_dinamico", index=False)
        print(f"\n[OK] {ruta_xl.name}")
    except Exception as e:
        print(f"\n[AVISO] Excel no escrito ({e})")
        tab.to_csv(DIR_OUT / f"test_lag_candidatos_{fecha}.csv", index=False)

    graficar(tab, tab_h, sub, col_m, mejor, piv,
             DIR_OUT / f"test_lag_calendario_{BANCO}_{fecha}.png")


# ── Gráfico ───────────────────────────────────────────────────────────────────
def graficar(tab, tab_h, sub, col_m, mejor, piv, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib no disponible")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.26})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c3c2b7")
        ax.tick_params(colors=C_TINTA2, labelsize=8.5)
        ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)

    fig.suptitle("¿Aporta el rezago de calendario información de escala nueva?",
                 fontsize=13, fontweight="bold", color=C_TINTA, y=0.98)
    fig.text(0.5, 0.945,
             "Lo que decide es la correlación PARCIAL controlando por la anchura "
             "que el modelo ya predice, no la simple.",
             ha="center", fontsize=9, color=C_TINTA2, style="italic")

    # Panel 1 · simple vs parcial
    ax = axes[0, 0]
    t = tab.sort_values("parcial_ancho")
    yp = np.arange(len(t))
    _color = lambda c: (C_COMP if c == "sigma_22d"
                        else C_REC if c.startswith("rec") else C_CAND)
    cols = [_color(c) for c in t["candidato"]]
    ax.barh(yp - 0.19, t["corr_simple"], height=0.34, color=cols, alpha=0.35, zorder=3)
    ax.barh(yp + 0.19, t["parcial_ancho"], height=0.34, color=cols, zorder=3)
    for i, (s, p) in enumerate(zip(t["corr_simple"], t["parcial_ancho"])):
        ax.text(s, i - 0.19, f" {s:+.2f}", va="center", fontsize=7.5, color=C_TINTA2)
        ax.text(p, i + 0.19, f" {p:+.2f}", va="center", fontsize=8, color=C_TINTA)
    ax.axvline(0, color=C_REF, lw=1.2)
    ax.set_yticks(yp); ax.set_yticklabels(t["candidato"], fontsize=8.5)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman vs |error| realizado", fontsize=9, color=C_TINTA2)
    ax.set_title("1 · Simple (pálido) vs parcial | anchura (sólido)\n"
                 f"naranja = calendario · verde = recencia pura · "
                 f"azul = {NOMBRE_COMPETIDOR or 'sin competidor'}",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    # Panel 2 · parcial con IC
    ax = axes[0, 1]
    t2 = tab.dropna(subset=["ic_lo"]).sort_values("parcial_ancho")
    yp2 = np.arange(len(t2))
    cols2 = [_color(c) for c in t2["candidato"]]
    ax.errorbar(t2["parcial_ancho"], yp2,
                xerr=[t2["parcial_ancho"] - t2["ic_lo"], t2["ic_hi"] - t2["parcial_ancho"]],
                fmt="none", ecolor=C_REF, elinewidth=1.4, capsize=4, zorder=3)
    ax.scatter(t2["parcial_ancho"], yp2, c=cols2, s=70, zorder=4)
    ax.axvline(0, color=C_REF, lw=1.2, ls="--")
    ax.set_yticks(yp2); ax.set_yticklabels(t2["candidato"], fontsize=8.5)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Parcial | anchura, con IC 90%", fontsize=9, color=C_TINTA2)
    ax.set_title("2 · ¿Sobrevive a la dependencia de los datos?\n"
                 "IC por bootstrap de bloques sobre fechas de origen",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    # Panel 3 · por horizonte
    ax = axes[1, 0]
    xh = np.arange(len(tab_h))
    ax.bar(xh - 0.19, tab_h["parcial_candidato"], width=0.36, color=C_CAND,
           zorder=3, label=mejor)
    ax.bar(xh + 0.19, tab_h["parcial_sigma"], width=0.36, color=C_COMP,
           zorder=3, label=NOMBRE_COMPETIDOR or "sin competidor")
    ax.axhline(0, color=C_REF, lw=1.2)
    ax.set_xticks(xh)
    ax.set_xticklabels(tab_h["tramo"], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Parcial | anchura", fontsize=9, color=C_TINTA2)
    ax.set_title("3 · Por tramo de horizonte\n"
                 "la disponibilidad de rezagos cae al alejarse",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=C_GRID, labelcolor=C_TINTA)

    # Panel 4 · rango dinámico intra-decil
    ax = axes[1, 1]
    xq = np.arange(len(piv))
    ax.bar(xq, piv.values, width=0.56, color=C_CAND, zorder=3)
    for i, v in enumerate(piv.values):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color=C_TINTA)
    ax.set_xticks(xq); ax.set_xticklabels([f"Q{i+1}" for i in xq], fontsize=9)
    ax.set_xlabel("Quintil del candidato, DENTRO del decil de anchura",
                  fontsize=9, color=C_TINTA2)
    ax.set_ylabel("Mediana del |error| (MM USD)", fontsize=9, color=C_TINTA2)
    ratio = piv.iloc[-1] / piv.iloc[0] if len(piv) >= 2 and piv.iloc[0] > 0 else np.nan
    ax.set_title(f"4 · Variación que el modelo se pierde: {ratio:.2f}x\n"
                 "filas que hoy reciben la misma banda y no deberían",
                 fontsize=10, fontweight="bold", color=C_TINTA, loc="left")

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OK] {ruta.name}")


if __name__ == "__main__":
    main()
