"""
step005_validar_sadf.py  (EWMA / Vol realizada + winsor opcional)
=======================
Valida el detector de régimen HMM (expanding window, sin leakage) antes de
integrarlo en la matriz de features.

DOS OPCIONES configurables en la cabecera:

  OPCIÓN 1 — fuente de la volatilidad que alimenta al HMM (FUENTE_VOLATILIDAD):
    "ewma"      → EWMA (RiskMetrics) con λ fijo (LAMBDA_EWMA=0.92, elegido por
                  grilla contra la vol realizada). No estima nada por bloque,
                  evita el arrastre IGARCH del GARCH que degeneraba el severo.
    "realizada" → volatilidad realizada cruda (rolling std de
                  VOL_REALIZADA_VENTANA días). Sin modelo. Pierde los primeros
                  (ventana-1) días del bloque por NaN (se descartan).

  OPCIÓN 2 — winsorización solo para estimación (WINSOR_ESTIMACION):
    True  → el HMM se ENTRENA sobre X capado a [P_INF, P_SUP] (acota leverage
            de outliers sobre medias/covarianzas), pero CLASIFICA sobre X real
            (un día extremo genuino puede seguir siendo severo).
    False → estándar, sin winsorización.

Único output: gráfico de evolución — cada panel re-etiqueta toda la historia
con el modelo (volatilidad elegida)+HMM entrenado hasta ese año.

Uso:
    python step005_validar_sadf.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

try:
    from hmmlearn import hmm as hmmlearn_hmm
    from sklearn.preprocessing import StandardScaler
    _HMM_OK = True
except ImportError:
    _HMM_OK = False
    print("AVISO: hmmlearn o sklearn no instalado.")

# ── Configuración ──────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"
RUTA_BCRP    = BASE_SISTEMA / "1. Data" / "Raw" / "series_bcrp.xlsx"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output"

BANCO              = "SISTEMA"
H_REF              = 2
N_ESTADOS          = 3   # 2 o 3. Con 2: solo calma/severo (sin moderado).

# Nombres y colores derivados de N_ESTADOS — un solo lugar para cambiar si
# se agrega/quita un estado, en vez de tocar cada print/gráfico por separado.
# Orden siempre ascendente por volatilidad (ver _fit_hmm: se ordena por
# det(covars_)), así que el último nombre/color SIEMPRE corresponde al
# estado más severo, sea cual sea N_ESTADOS.
if N_ESTADOS == 2:
    NOMBRES_ESTADOS = ["calma", "severo"]
    COLORES_ESTADOS = {0: "#4CAF50", 1: "#F44336"}            # verde, rojo
elif N_ESTADOS == 3:
    NOMBRES_ESTADOS = ["calma", "moderado", "severo"]
    COLORES_ESTADOS = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}  # verde, ámbar, rojo
else:
    # Fallback genérico si alguien prueba N_ESTADOS=4+ — no falla, pero sin
    # nombres propios (no se pidieron para ese caso).
    NOMBRES_ESTADOS = [f"estado_{i}" for i in range(N_ESTADOS)]
    import matplotlib.cm as _cm, matplotlib.colors as _mcolors
    _cmap = _cm.get_cmap("RdYlGn_r")
    COLORES_ESTADOS = {i: _mcolors.to_hex(_cmap(i / max(N_ESTADOS - 1, 1)))
                       for i in range(N_ESTADOS)}
    print(f"AVISO: N_ESTADOS={N_ESTADOS} fuera de {{2,3}} — usando nombres "
         f"genéricos {NOMBRES_ESTADOS}.")
    
HMM_INICIO         = "2019-07-01"
HMM_PRIMER_VENTANA = 3   # años del primer bloque in-sample
HMM_PASO_AVANCE    = 0.5 # años entre cortes sucesivos (expanding). Acepta
                         # fracciones — 0.5 = cortes cada 6 meses. Solo aplica
                         # cuando fechas_corte=None en hmm_evolucion (si el
                         # walk-forward pasa fechas_corte explícitas, esto se
                         # ignora — el paso queda determinado por esas fechas).

# ── OPCIÓN 1: fuente de la volatilidad que alimenta al HMM ──────────────────
#   "ewma"      → EWMA (RiskMetrics) con λ fijo (LAMBDA_EWMA). No estima nada.
#   "realizada" → volatilidad realizada cruda: rolling std de VOL_REALIZADA_VENTANA
#                 días, sin modelo. Pierde los primeros (ventana-1) días por NaN.
FUENTE_VOLATILIDAD      = "ewma"   # "ewma" | "realizada"
LAMBDA_EWMA             = 0.92     # λ del EWMA (estructural, elegido por grilla)
LAMBDA_INIT_VAR_VENTANA = 60       # días iniciales para sembrar la varianza EWMA
VOL_REALIZADA_VENTANA   = 20       # ventana (días) de la rolling std realizada

# ── OPCIÓN 3: estabilidad HMM — múltiples arranques + filtro de degeneración ──
# El algoritmo EM puede caer en mínimos locales donde uno o más estados pierden
# su auto-persistencia (diag(A) → 0), lo que colapsa esos estados a meros
# "pasos" hacia severo y distorsiona toda la clasificación de régimen.
# Solución: múltiples reinicios con semillas distintas; elegir el resultado que
# maximiza log-likelihood SUJETO A min(diag(A)) >= HMM_MIN_DIAG_TRANSMAT.
#
# HMM_N_STARTS       : número de reinicios con semillas distintas (>1 activa).
#                      1 = comportamiento original (solo random_state=42).
# HMM_MIN_DIAG_TRANSMAT : umbral mínimo de auto-persistencia por estado.
#                      Si ningún arranque lo cumple, se elige el menos degenerado
#                      y se loggea un WARNING con el diag(A) resultante.
# HMM_EXCLUIR_FOLDS_DEGENERADOS : si True, el fold queda marcado en el parquet
#                      (columna 'degenerado'=True) y se excluye de la estimación
#                      de ρ_s en step005_walk_forward_cv_4.py para evitar que
#                      correlaciones basura contaminen la simulación.
HMM_N_STARTS              = 20    # 1 = sin reinicios (original); ≥5 recomendado
HMM_MIN_DIAG_TRANSMAT     = 0.50  # mínimo de auto-persistencia por estado
HMM_EXCLUIR_FOLDS_DEGENERADOS = False  # marcar/excluir folds con diag<umbral
#   True  → las 2 dimensiones de X se capan a [P_INF, P_SUP] ANTES del scaler,
#           y el HMM se ENTRENA sobre esa versión capada (acota el leverage de
#           outliers sobre medias/covarianzas). La CLASIFICACIÓN (Viterbi) sigue
#           usando los valores reales sin capar → un día extremo genuino puede
#           seguir clasificándose como severo.
#   False → comportamiento estándar (sin winsorización).
WINSOR_ESTIMACION       = True
WINSOR_PCTL_INF         = 1.0      # percentil inferior de capado (%)
WINSOR_PCTL_SUP         = 99.0     # percentil superior de capado (%)

# ── Guardado de objetos para la simulación posterior (step006) ──────────────
#   True → al final de main(), guarda en DIR_OUTPUT:
#     - estados_regimen_hmm_<BANCO>.parquet : formato LARGO, una fila por
#       (año_corte, fecha). año_corte = año hasta el cual se entrenó ese fold.
#       Columnas: año_corte, fecha, estado, sigma, flujo, fuente_volatilidad,
#       winsor_estim. Se usa como FEATURE de XGBoost: para el fold de XGBoost
#       con corte en año Y, filtrar año_corte==Y (o el más reciente ≤ Y) y
#       merge por 'fecha' contra matriz_features — evita leakage porque usa
#       el HMM que solo vio datos hasta ese mismo punto.
#     - transmat_hmm_<BANCO>.parquet : una fila por año_corte, matriz 3x3
#       reordenada (calma/moderado/severo) aplanada en columnas p00..p22.
#       Se usa para SIMULAR paths (step006): elegir la fila del año_corte más
#       reciente disponible y reconstruir con .reshape(3,3).
GUARDAR_OBJETOS_SIMULACION = True

EPISODIOS = [
    ("2020-03-01", "2020-12-31", "COVID-19"),
    ("2021-03-01", "2021-08-31", "Elecciones 2021"),
]

# ── Carga de datos ─────────────────────────────────────────────────────────────

def cargar_datos(banco: str = BANCO, ruta_matriz: Path = RUTA_MATRIZ):
    """
    Serie de flujo neto diario de una entidad, indexada por la fecha de
    REALIZACIÓN (fecha_t + H_REF días hábiles), no por el origen.

    Los dos parámetros existen para que step005_walk_forward_cv_*.py pueda
    generar los regímenes de cada entidad en la misma corrida, sin tocar las
    constantes del módulo:

      - banco       : con PARTICIONES=True las entidades son FOCO_<PART> y
                      RESTO_<PART>, no solo SISTEMA.
      - ruta_matriz : con PARTICIONES=True la matriz es
                      matriz_features_particiones_<part>.parquet. Filtrar
                      banco="FOCO_BBVA" sobre la matriz v1 no falla: devuelve
                      un DataFrame vacío y el error aparece mucho después.

    Los valores por defecto son las constantes del módulo, así que main() y el
    uso standalone desde Spyder no cambian en nada.
    """
    df = pd.read_parquet(
        ruta_matriz,
        columns=["fecha_t", "banco", "h", "target"],
        filters=[("banco", "==", banco), ("h", "==", H_REF)],
    )
    if df.empty:
        raise ValueError(
            f"No hay filas para banco={banco!r} con h={H_REF} en "
            f"{ruta_matriz.name}. Si la entidad es FOCO_*/RESTO_*, la matriz "
            f"debe ser la generada por step001_build_feature_matrix_v2.py con "
            f"particion_activa != None.")
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values("fecha_t").reset_index(drop=True)
    df["fecha"] = df["fecha_t"] + pd.tseries.offsets.BusinessDay(H_REF)
    df = df.set_index("fecha").sort_index()
    print(f"  {banco} | {len(df):,} obs | {df.index.min().date()} → {df.index.max().date()}")
    return df["target"].dropna()


def cargar_embig():
    if not RUTA_BCRP.exists():
        return pd.Series(dtype=float, name="EMBI_PERU")
    try:
        raw = pd.read_excel(RUTA_BCRP, sheet_name="EMBIG", header=None)
        header_row = next(
            (i for i, row in raw.iterrows()
             if any(str(v).strip().lower() in ("date", "fecha") for v in row if pd.notna(v))),
            5,
        )
        df = pd.read_excel(RUTA_BCRP, sheet_name="EMBIG", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        col_f = next((c for c in df.columns if c.lower() in ("date", "fecha")), df.columns[0])
        col_v = next((c for c in df.columns if c.lower() in ("valores", "value", "values")), df.columns[1])
        df = df[[col_f, col_v]].copy()
        df.columns = ["fecha", "EMBI_PERU"]
        _m = {"Ene":"Jan","Feb":"Feb","Mar":"Mar","Abr":"Apr","May":"May","Jun":"Jun",
              "Jul":"Jul","Ago":"Aug","Set":"Sep","Oct":"Oct","Nov":"Nov","Dic":"Dec"}
        def _parse(s):
            s = str(s).strip()
            for es, en in _m.items():
                s = s.replace(es, en)
            return pd.to_datetime(s, errors="coerce", dayfirst=True)
        df["fecha"] = df["fecha"].apply(_parse)
        df["EMBI_PERU"] = pd.to_numeric(
            df["EMBI_PERU"].astype(str).str.replace(",", "."), errors="coerce"
        )
        return df.dropna(subset=["fecha"]).set_index("fecha").sort_index()["EMBI_PERU"]
    except Exception as e:
        print(f"  AVISO EMBIG: {e}")
        return pd.Series(dtype=float, name="EMBI_PERU")


# ── Volatilidad EWMA (RiskMetrics) con λ fijo ───────────────────────────────

def _ewma_vol(arr: np.ndarray, lam: float = LAMBDA_EWMA,
              init_ventana: int = LAMBDA_INIT_VAR_VENTANA) -> np.ndarray:
    """
    Volatilidad EWMA con λ FIJO (no estimado). Recursión causal:
        σ²_t = (1-λ)·x²_{t-1} + λ·σ²_{t-1}
    Equivale a un GARCH(1,1) con ω=0, α=1-λ, β=λ.

    La varianza inicial se siembra con la varianza muestral de los primeros
    `init_ventana` días. No estima nada → no hay inestabilidad de reestimación
    entre folds; σ depende solo de λ y de los datos. Devuelve σ_t en las
    unidades originales del flujo (no estandarizadas).
    """
    x = arr.astype(float)
    n = len(x)
    if n < 2:
        return np.full(n, float(np.std(x)) if n else 1.0)
    s2 = np.empty(n)
    seed = float(np.var(x[:min(init_ventana, n)]))
    s2[0] = seed if seed > 0 else 1.0
    for t in range(1, n):
        s2[t] = (1.0 - lam) * x[t-1]**2 + lam * s2[t-1]
    return np.sqrt(np.maximum(s2, 0.0))


def _ewma_last_s2(arr: np.ndarray, lam: float = LAMBDA_EWMA,
                  init_ventana: int = LAMBDA_INIT_VAR_VENTANA) -> float:
    """
    Propaga la recursión EWMA por arr y retorna el σ² del paso siguiente
    (σ²_{n+1}). Útil para proyección causal un paso adelante en producción.
    """
    x = arr.astype(float)
    n = len(x)
    if n < 2:
        return float(np.var(x)) if n else 1.0
    seed = float(np.var(x[:min(init_ventana, n)]))
    s2 = seed if seed > 0 else 1.0
    for t in range(1, n):
        s2 = (1.0 - lam) * x[t-1]**2 + lam * s2
    return (1.0 - lam) * x[-1]**2 + lam * s2   # σ²_{n+1}


def _vol_realizada(arr: np.ndarray, ventana: int = VOL_REALIZADA_VENTANA) -> np.ndarray:
    """
    Volatilidad realizada cruda: rolling std (causal, usa los últimos `ventana`
    días) sin ningún modelo. Devuelve un array del mismo largo que arr con NaN
    en los primeros (ventana-1) días, que el caller debe descartar.
    """
    s = pd.Series(arr.astype(float))
    return s.rolling(ventana).std().values


def _winsorizar(X: np.ndarray, p_inf: float, p_sup: float) -> np.ndarray:
    """
    Capa cada columna de X a sus percentiles [p_inf, p_sup]. Solo se usa para
    construir la matriz con la que se ESTIMAN los parámetros del HMM; los
    valores reales (sin capar) se siguen usando para clasificar con Viterbi.
    """
    X_w = X.copy()
    for col in range(X.shape[1]):
        lo, hi = np.percentile(X[:, col], [p_inf, p_sup])
        X_w[:, col] = np.clip(X[:, col], lo, hi)
    return X_w


# ── HMM ───────────────────────────────────────────────────────────────────────

def _fit_hmm(X: np.ndarray):
    """
    Ajusta GaussianHMM(N_ESTADOS) con múltiples reinicios (HMM_N_STARTS) para
    evitar el colapso de estados (state collapse problem del EM de Baum-Welch).

    El colapso ocurre cuando el EM converge a un mínimo local donde uno o más
    estados pierden auto-persistencia (diag(A) → 0), volviéndose estados
    transitorios sin estructura real. Se manifiesta como severo=40%+ con
    diag(A)=[0.0, 0.0, 0.9xx] — toda la clasificación pasa a través de esos
    estados degenerados.

    Estrategia: HMM_N_STARTS reinicios con semillas distintas. Se selecciona el
    resultado con mayor log-likelihood SUJETO A min(diag(A_reord)) >= umbral.
    Si ningún arranque cumple el umbral, se elige el menos degenerado con aviso.
    """
    scaler = StandardScaler()
    if WINSOR_ESTIMACION:
        X_fit = _winsorizar(X, WINSOR_PCTL_INF, WINSOR_PCTL_SUP)
    else:
        X_fit = X
    X_scaled = scaler.fit_transform(X_fit)

    mejor_modelo  = None
    mejor_scaler  = scaler
    mejor_ss      = None
    mejor_loglik  = -np.inf
    mejor_diag_ok = False   # ¿cumple el umbral de no-degeneración?

    for intento in range(max(1, HMM_N_STARTS)):
        seed = 42 + intento * 17   # semillas reproducibles pero distintas
        try:
            m = hmmlearn_hmm.GaussianHMM(
                n_components=N_ESTADOS, covariance_type="full",
                n_iter=1000, random_state=seed,
            )
            m.fit(X_scaled)
            ss_i = np.argsort([np.linalg.det(m.covars_[s]) for s in range(N_ESTADOS)])
            A_i  = m.transmat_[np.ix_(ss_i, ss_i)]
            diag_min = float(np.diag(A_i).min())
            try:
                loglik = float(m.score(X_scaled))
            except Exception:
                loglik = -np.inf

            diag_ok = diag_min >= HMM_MIN_DIAG_TRANSMAT

            # Preferir: (1) cumple umbral + mayor LL; (2) si ninguno cumple,
            # el de mayor diag_min entre los degenerados.
            if diag_ok and (not mejor_diag_ok or loglik > mejor_loglik):
                mejor_modelo, mejor_ss = m, ss_i
                mejor_loglik  = loglik
                mejor_diag_ok = True
            elif not mejor_diag_ok and diag_min > float(
                np.diag(mejor_modelo.transmat_[np.ix_(mejor_ss, mejor_ss)]).min()
                if mejor_modelo is not None else -1
            ):
                mejor_modelo, mejor_ss = m, ss_i
                mejor_loglik  = loglik

        except Exception as _e:
            continue

    if mejor_modelo is None:
        # Fallback absoluto si todos los intentos fallaron
        mejor_modelo = hmmlearn_hmm.GaussianHMM(
            n_components=N_ESTADOS, covariance_type="full",
            n_iter=1000, random_state=42,
        )
        mejor_modelo.fit(X_scaled)
        mejor_ss = np.argsort(
            [np.linalg.det(mejor_modelo.covars_[s]) for s in range(N_ESTADOS)])

    A_final   = mejor_modelo.transmat_[np.ix_(mejor_ss, mejor_ss)]
    diag_final = np.diag(A_final)
    if not mejor_diag_ok:
        print(f"    ⚠ ADVERTENCIA: ningún arranque superó umbral "
              f"HMM_MIN_DIAG_TRANSMAT={HMM_MIN_DIAG_TRANSMAT}. "
              f"diag(A) final={diag_final.round(3).tolist()} "
              f"— fold marcado como degenerado.")

    return mejor_modelo, mejor_scaler, mejor_ss, mejor_diag_ok


def _predecir(modelo, scaler, ss, X: np.ndarray) -> np.ndarray:
    """
    Clasifica con Viterbi usando los valores REALES de X (sin winsorizar).
    La winsorización (si está activa) solo afecta la estimación de parámetros
    en _fit_hmm, no la clasificación: un día extremo genuino puede clasificarse
    como severo.
    """
    raw = modelo.predict(scaler.transform(X))
    mapa = {ss[i]: i for i in range(N_ESTADOS)}
    return np.array([mapa[e] for e in raw])


def _transmat_reordenada(modelo, ss) -> np.ndarray:
    """
    Devuelve transmat_ reordenada según orden ascendente de volatilidad —
    0=calma, ..., N_ESTADOS-1=severo (mismo orden 'ss' que usa _predecir
    para las etiquetas; ver NOMBRES_ESTADOS). Sin este reordenamiento,
    transmat_ queda en el orden interno arbitrario de hmmlearn. Necesaria
    para guardar la matriz de transición que alimenta la simulación de
    paths (step006).
    """
    return modelo.transmat_[np.ix_(ss, ss)]


# ── Evolución expanding ────────────────────────────────────────────────────────

def hmm_evolucion(
    flujo: pd.Series,
    primer_ventana: float = HMM_PRIMER_VENTANA,
    fechas_corte: list | None = None,
    paso_avance: float = HMM_PASO_AVANCE,
) -> dict:
    """
    Entrena y re-etiqueta bloques expanding del HMM, uno por cada fecha de corte.

    La lógica ANTERIOR usaba años enteros como cortes (año_corte = 2021, 2022…),
    lo que producía tres problemas:
      1. Si primer_ventana=3.5, año_fin_prim=2021.5 (float) → el merge del
         walk-forward buscaba enteros y no lo encontraba.
      2. No podía generar cortes a mitad de año (train_end=2021-06-30).
      3. HMM_INICIO era ignorado; usaba flujo.index.year.min().

    La lógica NUEVA trabaja con fechas exactas (pd.Timestamp):
      - Si fechas_corte se pasa explícitamente (lista de Timestamps), las usa
        directamente. Esto permite alinearlas 1:1 con los train_end del
        walk-forward y así garantizar que siempre exista un bloque HMM válido
        para cada fold (sin importar si es mid-año).
      - Si fechas_corte=None, las genera internamente a partir de HMM_INICIO
        y primer_ventana (en años), avanzando cada paso_avance años desde esa
        fecha (por defecto 1 año; acepta fracciones, p.ej. 0.5 = cada 6 meses).

    Las claves del dict devuelto son pd.Timestamp (no int), para que el merge
    en reemplazar_regimen_fold con train_end (también Timestamp) sea exacto.

    Parámetros
    ----------
    flujo         : Serie con el flujo neto diario, índice DatetimeIndex.
    primer_ventana: años del primer bloque (usado solo si fechas_corte=None).
    fechas_corte  : lista de pd.Timestamp con los cortes exactos. Si se pasa,
                    primer_ventana y paso_avance se ignoran. Típicamente =
                    train_end de cada fold del walk-forward.
    paso_avance   : años entre cortes sucesivos, tras el primero (usado solo
                    si fechas_corte=None). Acepta fracciones — 0.5 avanza
                    cada 6 meses, 0.25 cada 3 meses. Mismo mecanismo de
                    conversión fraccionaria que primer_ventana (vía
                    relativedelta, sin el error de 0.5*365=182.5 días).

    Devuelve
    --------
    dict {pd.Timestamp: 9-tupla} — una entrada por corte procesado.
    """
    flujo = flujo.sort_index()

    # ── Determinar fechas de corte ────────────────────────────────────────────
    inicio = pd.Timestamp(HMM_INICIO)

    if fechas_corte is not None:
        # Usar las fechas pasadas directamente (modo integrado con walk-forward)
        cortes = sorted(pd.Timestamp(f) for f in fechas_corte)
    else:
        # Generar internamente: primer corte = inicio + primer_ventana años,
        # luego avanzar de año en año hasta el final de los datos disponibles.
        # Se usa DateOffset para manejar correctamente años con distinta cantidad
        # de días (sin el error de 3.5 * 365 = 1277.5 días).
        from dateutil.relativedelta import relativedelta
        años_int  = int(primer_ventana)
        meses_fra = round((primer_ventana - años_int) * 12)
        primer_corte = inicio + relativedelta(years=años_int, months=meses_fra)

        # Cortes desde primer_corte hasta el fin de los datos, avanzando
        # paso_avance años cada vez (mismo mecanismo fraccionario que arriba
        # — evita el error de paso_avance*365 con años bisiestos/no enteros).
        paso_años_int  = int(paso_avance)
        paso_meses_fra = round((paso_avance - paso_años_int) * 12)
        if paso_años_int == 0 and paso_meses_fra == 0:
            raise ValueError(f"paso_avance={paso_avance} produce un paso de "
                            f"0 meses — revisa HMM_PASO_AVANCE (debe ser > 0).")
        paso_delta = relativedelta(years=paso_años_int, months=paso_meses_fra)

        cortes = []
        c = primer_corte
        while c <= flujo.index.max():
            cortes.append(c)
            c = c + paso_delta

    resultados = {}
    desc_vol = (f"EWMA(λ={LAMBDA_EWMA})" if FUENTE_VOLATILIDAD == "ewma"
                else f"VolReal({VOL_REALIZADA_VENTANA}d)")
    _paso_info = (f"paso_avance={paso_avance}a" if fechas_corte is None
                  else "paso_avance=n/a (fechas_corte explícitas)")
    print(f"\n  [EVOLUCIÓN] {desc_vol}+HMM por bloque | "
          f"HMM_INICIO={HMM_INICIO} | "
          f"primer_ventana={primer_ventana}a | {_paso_info} | "
          f"winsor={WINSOR_ESTIMACION}")

    def _bloque(flujo_bloque, label):
        if len(flujo_bloque) < 60:
            print(f"    {label}: bloque muy corto ({len(flujo_bloque)} obs)")
            return None

        if FUENTE_VOLATILIDAD == "ewma":
            sigma   = _ewma_vol(flujo_bloque.values, LAMBDA_EWMA)
            idx_v   = flujo_bloque.index
            flujo_v = flujo_bloque.values
        elif FUENTE_VOLATILIDAD == "realizada":
            sigma_raw = _vol_realizada(flujo_bloque.values, VOL_REALIZADA_VENTANA)
            valido    = ~np.isnan(sigma_raw)
            sigma     = sigma_raw[valido]
            idx_v     = flujo_bloque.index[valido]
            flujo_v   = flujo_bloque.values[valido]
            if len(sigma) < 60:
                print(f"    {label}: tras descartar NaN, bloque muy corto "
                      f"({len(sigma)} obs)")
                return None
        else:
            raise ValueError(f"FUENTE_VOLATILIDAD desconocida: {FUENTE_VOLATILIDAD}")

        X = np.column_stack([flujo_v, sigma])
        modelo, scaler, ss, diag_ok = _fit_hmm(X)
        estados = _predecir(modelo, scaler, ss, X)
        A_reord = _transmat_reordenada(modelo, ss)
        pct    = (estados == N_ESTADOS - 1).mean() * 100
        last_s2 = float(_ewma_last_s2(flujo_v, LAMBDA_EWMA))
        flag_deg = " ⚠[DEGENERADO]" if not diag_ok else ""
        print(f"    hasta {label}: {len(flujo_v):,} obs | "
              f"{desc_vol} | {NOMBRES_ESTADOS[-1]}={pct:.0f}% | "
              f"diag(A)={np.diag(A_reord).round(3).tolist()}{flag_deg}")
        return (idx_v, estados, pd.Series(sigma, index=idx_v),
                A_reord, modelo, scaler, ss, last_s2, diag_ok)

    for corte in cortes:
        # Datos disponibles desde HMM_INICIO hasta la fecha de corte inclusive
        bloque = flujo[(flujo.index >= inicio) & (flujo.index <= corte)]
        label  = corte.strftime("%Y-%m-%d")
        try:
            res = _bloque(bloque, label)
            if res:
                resultados[corte] = res   # clave = pd.Timestamp exacto
        except Exception as e:
            print(f"    {label}: falló — {e}")

    return resultados


# ── Gráfico de evolución ───────────────────────────────────────────────────────

def graficar_evolucion(flujo: pd.Series, evol: dict, embig: pd.Series = None) -> None:
    años = sorted(evol.keys())
    n    = len(años)
    if n == 0:
        return
    # Convertir claves a labels legibles (soporta Timestamp y int legacy)
    def _label(k):
        return k.strftime("%Y-%m-%d") if hasattr(k, "strftime") else str(k)

    tiene_embig = embig is not None and not embig.empty
    n_rows   = (1 if tiene_embig else 0) + 1 + n
    h_ratios = ([1.2] if tiene_embig else []) + [2.0] + [1.0] * n

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(19, 2.0 + 2.5 * n),
        sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.05},
    )
    _desc_vol_titulo = (f"EWMA(λ={LAMBDA_EWMA})" if FUENTE_VOLATILIDAD == "ewma"
                        else f"VolReal({VOL_REALIZADA_VENTANA}d)")
    fig.suptitle(
        f"Evolución del HMM — {BANCO}  |  {N_ESTADOS} estados"
        f"{'  | winsor estim.' if WINSOR_ESTIMACION else ''}\n"
        f"Cada panel re-etiqueta TODA la historia con el modelo "
        f"{_desc_vol_titulo}+HMM entrenado hasta ese año",
        fontsize=12, fontweight="bold", y=0.995,
    )

    colores_hmm = COLORES_ESTADOS
    sig = 0   # índice de panel siguiente

    # Panel EMBIG
    if tiene_embig:
        ax = axes[sig]; sig += 1
        ev = embig.reindex(flujo.index.union(embig.index)).sort_index()
        ev = ev[ev.index >= flujo.index.min()]
        ax.plot(ev.index, ev.values, color="#6A1B9A", lw=0.8)
        ax.fill_between(ev.index, ev.values, alpha=0.12, color="#6A1B9A")
        ax.set_ylabel("EMBIG\n(pbs)", fontsize=8)
        ax.set_title("EMBIG Perú — riesgo país (pbs)", fontsize=9)
        for ini, fin, _ in EPISODIOS:
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), alpha=0.10, color="navy", zorder=0)

    # Panel flujo neto
    ax = axes[sig]; sig += 1
    cols = np.where(flujo.values >= 0, "#2196F3", "#F44336")
    ax.bar(flujo.index, flujo.values, color=cols, alpha=0.65, width=1.2)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_ylabel("Flujo neto", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))
    ax.set_title("Flujo neto diario (azul=entradas, rojo=salidas netas)", fontsize=9)
    for ini, fin, etiqueta in EPISODIOS:
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), alpha=0.10, color="navy", zorder=0)
        mid  = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
        ymax = ax.get_ylim()[1]
        ax.text(mid, ymax * 0.88, etiqueta, ha="center", fontsize=8, color="#1A237E",
                bbox=dict(boxstyle="round,pad=0.2", fc="lavender", alpha=0.8))

    # Paneles HMM por año
    for i, año in enumerate(años):
        ax_i = axes[sig + i]
        idx_a, est_a, sigma_a, _A_año, *_modelo_tuple = evol[año]   # _A_año, modelo no se usan en el plot

        for estado, color in colores_hmm.items():
            for f in idx_a[est_a == estado]:
                ax_i.axvspan(f, f + pd.offsets.BusinessDay(1),
                             alpha=0.55, color=color, linewidth=0)

        sig_n = (sigma_a - sigma_a.mean()) / (sigma_a.std() + 1e-12)
        ax_i.plot(sig_n.index, sig_n.values, color="k", lw=0.5, alpha=0.5)

        for ini, fin, _ in EPISODIOS:
            if pd.Timestamp(ini) <= idx_a.max():
                ax_i.axvspan(pd.Timestamp(ini),
                             min(pd.Timestamp(fin), idx_a.max()),
                             alpha=0.12, color="navy", zorder=0)

        pct = (est_a == N_ESTADOS - 1).mean() * 100
        ax_i.set_ylabel(str(año), fontsize=8, rotation=0, labelpad=35, va="center")
        ax_i.set_title(f"Modelo entrenado hasta {_label(año)}  ({len(idx_a):,} obs)  —  "
                       f"{NOMBRES_ESTADOS[-1]}={pct:.0f}%", fontsize=8, loc="left", pad=2)
        ax_i.set_yticks([])

        if i == 0:
            parches = [mpatches.Patch(color=COLORES_ESTADOS[s], label=nombre.capitalize())
                       for s, nombre in enumerate(NOMBRES_ESTADOS)]
            ax_i.legend(handles=parches, fontsize=7, loc="upper left", ncol=N_ESTADOS)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.93, hspace=0.05)
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    ruta = DIR_OUTPUT / f"evolucion_hmm_{BANCO}.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print(f"\n  Gráfico guardado: {ruta}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def guardar_objetos_simulacion(evol: dict, flujo: pd.Series, banco: str = BANCO,
                               dir_output: Path = DIR_OUTPUT) -> None:
    """
    Guarda, por fold, los objetos necesarios para:
      (a) usar el estado de régimen como FEATURE de XGBoost (formato largo,
          mergeable por fecha, filtrando por año_corte), y
      (b) usar la matriz de transición para SIMULAR paths (step006), y
      (c) [NUEVO] clasificar el período de VALIDACIÓN con los parámetros del
          fold de TRAIN (sin re-entrenar) para estimar ρ_s sin leakage.
          Guarda un pickle por fold: modelo_hmm_{banco}_{año_corte}.pkl
          con (modelo, scaler, ss, last_s2, LAMBDA_EWMA). La semilla last_s2
          (σ²_{n+1} del último día de train) permite continuar la recursión
          EWMA de forma causal hacia el período de validación.

    No es leakage para (a) siempre que el fold de XGBoost use año_corte ≤ año
    de corte de su propio train (ver comentario junto a GUARDAR_OBJETOS_SIMULACION).
    """
    import pickle as _pkl
    filas_estados = []
    filas_transmat = []

    for corte_ts, tupla in evol.items():
        # corte_ts es pd.Timestamp (nuevo) o int (compatibilidad legacy)
        # año_corte en el parquet se guarda como string ISO para preservar
        # la fecha exacta de corte (evita la pérdida de información mid-año
        # que ocurría con el esquema de entero año).
        corte_str = corte_ts.strftime("%Y-%m-%d") if hasattr(corte_ts, "strftime") else str(corte_ts)

        idx_a, est_a, sigma_a, A_a = tupla[0], tupla[1], tupla[2], tupla[3]
        modelo_a = tupla[4] if len(tupla) > 4 else None
        scaler_a = tupla[5] if len(tupla) > 5 else None
        ss_a     = tupla[6] if len(tupla) > 6 else None
        last_s2  = tupla[7] if len(tupla) > 7 else None
        diag_ok  = bool(tupla[8]) if len(tupla) > 8 else True

        # [NUEVO] Guardar pickle del modelo HMM por fold
        if modelo_a is not None:
            ruta_pkl = dir_output / f"modelo_hmm_{banco}_{corte_str}.pkl"
            with open(ruta_pkl, "wb") as f:
                _pkl.dump({
                    "modelo":      modelo_a,
                    "scaler":      scaler_a,
                    "ss":          ss_a,
                    "last_s2":     last_s2,
                    "lambda_ewma": LAMBDA_EWMA,
                    "diag_ok":     diag_ok,    # False = fold degenerado
                    "corte":       corte_str,  # fecha exacta del corte
                }, f, protocol=4)

        flujo_a = flujo.reindex(idx_a)
        for fecha, estado, sig, fl in zip(idx_a, est_a, sigma_a.values, flujo_a.values):
            filas_estados.append({
                "año_corte":         corte_str,  # string ISO "YYYY-MM-DD"
                "fecha":             fecha,
                "estado":            int(estado),
                "sigma":             float(sig),
                "flujo":             float(fl),
                "fuente_volatilidad": FUENTE_VOLATILIDAD,
                "winsor_estim":      WINSOR_ESTIMACION,
                "degenerado":        not diag_ok,
            })
        fila_A = {"año_corte": corte_str}
        for i in range(N_ESTADOS):
            for j in range(N_ESTADOS):
                fila_A[f"p{i}{j}"] = float(A_a[i, j])
        filas_transmat.append(fila_A)

    df_estados = pd.DataFrame(filas_estados)
    df_transmat = pd.DataFrame(filas_transmat)

    dir_output.mkdir(parents=True, exist_ok=True)
    ruta_estados  = dir_output / f"estados_regimen_hmm_{banco}.parquet"
    ruta_transmat = dir_output / f"transmat_hmm_{banco}.parquet"
    df_estados.to_parquet(ruta_estados, index=False)
    df_transmat.to_parquet(ruta_transmat, index=False)

    print(f"\n  [GUARDADO] {ruta_estados.name}: {len(df_estados):,} filas "
          f"({df_estados['año_corte'].nunique()} folds)")
    print(f"  [GUARDADO] {ruta_transmat.name}: {len(df_transmat)} filas "
          f"(1 por fold)")
    n_pkls = sum(1 for f in dir_output.glob(f"modelo_hmm_{banco}_*.pkl"))
    print(f"  [GUARDADO] {n_pkls} pickles HMM (modelo_hmm_{banco}_<año>.pkl)")

def cargar_modelo_hmm_fold(banco: str, año_corte: int,
                           dir_output: Path = DIR_OUTPUT) -> dict:
    """
    Carga el pickle del modelo HMM del fold indicado.
    Devuelve dict con claves: modelo, scaler, ss, last_s2, lambda_ewma.
    Usado en step005_walk_forward_cv_4.py para clasificar VAL sin re-entrenar.
    """
    import pickle as _pkl
    ruta = dir_output / f"modelo_hmm_{banco}_{año_corte}.pkl"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No hay pickle HMM para banco={banco}, año_corte={año_corte} "            f"en {dir_output}. Corre step005_validar_hmm_v6.py con "            f"GUARDAR_OBJETOS_SIMULACION=True primero.")
    with open(ruta, "rb") as f:
        return _pkl.load(f)




def cargar_estados_regimen(banco: str = BANCO, año_corte: int = None,
                           dir_output: Path = DIR_OUTPUT) -> pd.DataFrame:
    """
    Carga estados_regimen_hmm_<banco>.parquet. Si año_corte se especifica,
    filtra a ese fold (lo típico para mergear con un fold de XGBoost del mismo
    corte). Si año_corte=None, devuelve todos los folds (formato largo).
    """
    ruta = dir_output / f"estados_regimen_hmm_{banco}.parquet"
    df = pd.read_parquet(ruta)
    if año_corte is not None:
        df = df[df["año_corte"] == año_corte]
    return df


def cargar_transmat(banco: str = BANCO, año_corte: int = None,
                    dir_output: Path = DIR_OUTPUT) -> np.ndarray:
    """
    Carga transmat_hmm_<banco>.parquet y reconstruye la matriz N_ESTADOS x
    N_ESTADOS del fold pedido. Si año_corte=None, usa el fold más reciente
    (el de mayor año_corte) — el típico para simular paths hacia adelante
    desde hoy.
    """
    ruta = dir_output / f"transmat_hmm_{banco}.parquet"
    df = pd.read_parquet(ruta)
    if año_corte is None:
        año_corte = df["año_corte"].max()
    fila = df[df["año_corte"] == año_corte]
    if fila.empty:
        raise ValueError(f"No hay transmat guardada para año_corte={año_corte}")
    cols = [f"p{i}{j}" for i in range(N_ESTADOS) for j in range(N_ESTADOS)]
    return fila.iloc[0][cols].values.astype(float).reshape(N_ESTADOS, N_ESTADOS)


def main():
    if not _HMM_OK:
        print("ERROR: instalar hmmlearn y sklearn antes de ejecutar.")
        return
    flujo = cargar_datos()
    flujo_hmm = flujo[flujo.index >= HMM_INICIO]   # mismo corte para calcular Y graficar
    evol  = hmm_evolucion(flujo_hmm)
    if GUARDAR_OBJETOS_SIMULACION:
        guardar_objetos_simulacion(evol, flujo)   # flujo completo: reindex(idx_a) es seguro
    embig = cargar_embig()
    graficar_evolucion(flujo_hmm, evol, embig=embig)


if __name__ == "__main__":
    main()
