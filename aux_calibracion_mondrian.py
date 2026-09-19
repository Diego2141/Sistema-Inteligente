"""
Calibración conformal Mondrian por posición en el mes, con validación
leave-one-fold-out. Lee los parquets de step005 y no re-entrena nada.

Qué corrige
-----------
El modelo cubre 94.3% en los 16 días centrales del mes y 42.4% en el último
día hábil. La media marginal disimula ambos. Cualquier corrección global es
ciega a eso por construcción, porque garantiza P(y en C) = 1-alpha y no
P(y en C | posición del mes) = 1-alpha.

Cómo
----
Para cada fila i = (fold, h, fecha_t) con fecha_th = fecha_t + h días hábiles:

    w_lo = q50 - q05          s_lo = (q50 - y) / w_lo
    w_hi = q95 - q50          s_hi = (y - q50) / w_hi

s_lo > 1 equivale EXACTAMENTE a que y cruzó q05, así que el percentil 95 de
s_lo dentro de una categoría es el factor por el que hay que multiplicar esa
semi-amplitud para dejar la cola en 5%. El intervalo calibrado es

    q05_cal = q50 - k_lo(kappa) * w_lo        q95_cal = q50 + k_hi(kappa) * w_hi

y como k > 0 y w > 0, nunca se cruza con q50.

El factor se descompone en nivel x forma:

    k_f(kappa) = k_f^marg  x  [ k_pool(kappa) / k_pool^marg ]

La FORMA se agrupa entre folds, porque con ~10 fechas por categoría y fold no
alcanza para un percentil 95.

El NIVEL admite tres fuentes, y la elección resultó ser decisiva:

  test_previos  el TEST cuyo resultado ya se conoce al empezar a proyectar este
                período. Mide directamente lo que interesa: cuánto ancho faltó
                fuera de muestra.
  val_propio    el VAL del propio fold. Fue el primer diseño y está sesgado: en
                VAL el modelo SOBRE-cubre (k de 0.89, 0.70, 0.60 y 0.43 por
                fold) mientras en TEST sub-cubre (1.43, 0.98, 1.12, 1.34). Se
                aplicó 0.43 donde hacía falta 1.34, de modo que el nivel
                estrechaba incluso las categorías que pedían ensancharse:
                importaba la brecha VAL-TEST en vez de absorberla.
  sin nivel     nivel 1. Conserva la calibración global de step005 y solo
                redistribuye el ancho entre posiciones del mes.

El corte que separa "ya conocido" de "todavía no" es por FECHA DE RESULTADO,
no por número de fold. Las ventanas de TEST se solapan en fecha_th —cada una
se extiende hasta 75 días hábiles más allá de su último origen, invadiendo el
período del fold siguiente—, así que "el TEST del fold anterior" incluiría
resultados que aún no habrían ocurrido. El guardia de leakage lo verifica
sobre las dos fuentes y aborta si se viola.

Qué se calibra y qué no — decidido con aux_test_supuestos_mondrian.py
---------------------------------------------------------------------
  q05/q95  nivel por fold x forma por posición del mes, cada lado por separado
  q01/q99  solo nivel por fold. El percentil 99 del score exige 99 fechas por
           categoría y hay 27, así que la forma no es estimable ni validable
  q40/q60  sin tocar: su desvío es sesgo de ubicación, no de escala
  q50      sin tocar. La prueba 5 mostró que su sesgo cambia de signo entre
           folds por categoría pero deriva monótonamente en el tiempo
           (-0.045, -0.024, +0.103, +0.133 en 'resto'): es deriva temporal,
           no efecto de calendario, y corregirlo aquí sería atribuirle una
           causa equivocada

Por qué multiplicativo: el ancho crece con h (de ~800 a ~1250 MM) pero el
efecto de calendario no. Medido, el factor de 'inicio +2' dio 2.36/2.35/2.34/
2.35/2.37 en los cinco grupos de horizonte. Eso permite agrupar los 74
horizontes y multiplicar por ~74 los datos disponibles.

Fuentes de la FORMA — ver _conjunto_forma()
-------------------------------------------
  causal_todo  VAL y TEST con resultado ya conocido. Es la propuesta: mismo
               corte temporal que causal_val pero con la mitad de los datos
               viniendo de un régimen realista, y con ~43 fechas por categoría
               de borde en vez de ~27.
  causal_val   solo VAL previo. Hereda el mismo sesgo que tenía el nivel —VAL
               es más benigno que TEST— y por eso 'inicio +2' y 'cierre -3' no
               se movieron pese a recibir ensanchamiento.
  lofo         todos los VAL menos el propio. USA DATOS POSTERIORES: no es
               desplegable, sirve de cota superior.

Uso: ejecutar directamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración — espejo de step005_walk_forward_cv_4.py
# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"
DIR_CALIB    = BASE_SISTEMA / "2. Output" / "step006_mondrian"
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

EXPANDING     = False   # debe coincidir con la corrida que se quiere leer
AJUSTE_ARCTAN = True
BANCO         = "SISTEMA"
FECHA_TAG     = ""      # "" → la más reciente

J_BORDE     = 3      # días de cada extremo del mes con categoría propia
MIN_FECHAS  = 19     # ceil(0.95*(n+1)) <= n exige n >= 19
LAMBDA_ENC  = 0.70   # encogimiento hacia 1: r' = 1 + LAMBDA*(r-1)
K_MIN       = 0.50   # piso del factor; evita cruces y estrechamientos brutales
K_MAX       = 5.00   # techo; una estimación disparada no debe pasar
ALPHA_90    = 0.10   # para el Winkler de la banda 90%

TAUS_FORMA  = [0.05, 0.95]              # calendario + nivel
TAUS_NIVEL  = [0.01, 0.99]              # solo nivel
GUARDAR     = True

# Combinaciones a reportar: (fuente de FORMA, aplicar nivel, fuente de NIVEL).
# No es el producto cartesiano: la fuente del nivel ya quedó decidida
# —test_previos gana por 6.9 pp— así que se varía sobre todo la de la forma.
COMBINACIONES = [
    ("causal_todo", True, "test_previos"),   # la propuesta: forma de todo lo conocido
    ("causal_val",  True, "test_previos"),   # lo que ya se corrió, para comparar
    ("lofo",        True, "test_previos"),   # cota superior
    ("causal_val",  True, "val_propio"),     # referencia del sesgo del nivel
]

ORDEN = ["inicio +1", "inicio +2", "inicio +3", "resto del mes",
         "cierre -3", "cierre -2", "cierre -1"]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _dir_modo() -> Path:
    sfx = "_arctan" if AJUSTE_ARCTAN else ""
    return DIR_OUTPUT / f"fold_{'exp' if EXPANDING else 'roll'}{sfx}"


def _filtrar_tag(files: list[Path]) -> list[Path]:
    if FECHA_TAG:
        files = [f for f in files if FECHA_TAG in f.name]
    tags = sorted({f.stem.split("_")[-1] for f in files})
    if len(tags) > 1:
        files = [f for f in files if tags[-1] in f.name]
        print(f"[aviso] varias corridas {tags}; se usa {tags[-1]}")
    return files


def _cargar(dir_modo: Path, patron: str) -> pd.DataFrame:
    files = _filtrar_tag(sorted(dir_modo.glob(patron)))
    if not files:
        raise FileNotFoundError(f"No hay {patron} en {dir_modo}.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _adjuntar_fecha_th(d: pd.DataFrame) -> pd.DataFrame:
    """El scaffold de VAL de corridas anteriores no guardaba fecha_th."""
    if "fecha_th" in d.columns:
        return d
    if not RUTA_MATRIZ.exists():
        raise FileNotFoundError(
            f"VAL sin fecha_th y falta {RUTA_MATRIZ}. Vuelva a correr step005 "
            f"(ya la guarda) o corrija RUTA_MATRIZ."
        )
    print(f"[info] VAL sin fecha_th — se recupera de {RUTA_MATRIZ.name}")
    mp = (pd.read_parquet(RUTA_MATRIZ, columns=["fecha_t", "h", "fecha_th"])
          .drop_duplicates(subset=["fecha_t", "h"]))
    mp["fecha_t"] = pd.to_datetime(mp["fecha_t"])
    d = d.copy()
    d["fecha_t"] = pd.to_datetime(d["fecha_t"])
    d = d.merge(mp, on=["fecha_t", "h"], how="left")
    n_sin = int(d["fecha_th"].isna().sum())
    if n_sin:
        print(f"[aviso] {n_sin:,} filas sin fecha_th en la matriz — se descartan")
        d = d[d["fecha_th"].notna()]
    return d


def _mapa_posicion(fechas: np.ndarray) -> pd.DataFrame:
    """
    Posición de cada fecha dentro de su mes, sobre el calendario que forman las
    fechas observadas. Se construye UNA sola vez con la unión de VAL y TEST:
    hacerlo por separado podría numerar distinto los meses de solape.

    Los meses con menos de 15 fechas son bordes de ventana — el conteo estaría
    truncado y etiquetaría mal — y quedan sin categoría.
    """
    fe = pd.DataFrame({"fecha_th": pd.to_datetime(sorted(pd.unique(fechas)))})
    fe["mes"] = fe["fecha_th"].dt.to_period("M")
    cnt = fe.groupby("mes")["fecha_th"].transform("size")
    fe["p_ini"] = fe.groupby("mes").cumcount() + 1
    fe["p_fin"] = cnt - fe["p_ini"] + 1

    pos = np.where(
        cnt < 15, None,
        np.where(fe["p_ini"] <= J_BORDE,
                 "inicio +" + fe["p_ini"].astype(int).astype(str),
                 np.where(fe["p_fin"] <= J_BORDE,
                          "cierre -" + fe["p_fin"].astype(int).astype(str),
                          "resto del mes")))
    fe["pos"] = pos
    return fe[["fecha_th", "pos"]]


# ---------------------------------------------------------------------------
# Score y factores
# ---------------------------------------------------------------------------

def _score(d: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(score, semi-amplitud, nivel del percentil a estimar) para el cuantil tau."""
    col = f"q{int(tau * 100):02d}"
    if tau < 0.5:
        w = (d["q50"] - d[col]).clip(lower=1.0).to_numpy(float)
        return ((d["q50"] - d["target"]).to_numpy(float) / w, w, 1.0 - tau)
    w = (d[col] - d["q50"]).clip(lower=1.0).to_numpy(float)
    return ((d["target"] - d["q50"]).to_numpy(float) / w, w, tau)


def _k_conformal(s: np.ndarray, nivel: float) -> float:
    """Estadístico de orden ceil((n+1)*nivel) — corrección de muestra finita."""
    s = np.sort(np.asarray(s, dtype=float))
    n = s.size
    if n == 0:
        return np.nan
    return float(s[min(int(np.ceil((n + 1) * nivel)) - 1, n - 1)])


def _nivel_marginal(fuentes: list[pd.DataFrame], tau: float) -> float:
    """
    Nivel marginal para el cuantil tau: la MEDIANA del k marginal de cada
    fuente. Con una sola fuente (el VAL del propio fold) es simplemente su k.
    Con varias —los TEST de los folds anteriores— la mediana amortigua que un
    semestre atípico fije el nivel de todo el siguiente.
    """
    ks = []
    for src in fuentes:
        if len(src) == 0:
            continue
        s, _, niv = _score(src, tau)
        k = _k_conformal(s, niv)
        if np.isfinite(k) and k > 0:
            ks.append(k)
    return float(np.median(ks)) if ks else 1.0


def estimar_factores(val_forma: pd.DataFrame,
                     fuentes_nivel: list[pd.DataFrame],
                     log: list,
                     aplicar_nivel: bool = True) -> dict:
    """
    Devuelve {tau: {pos: k}} más {tau: "__marg__": k} para los que no llevan
    forma. val_forma alimenta la FORMA (agrupada entre folds) y fuentes_nivel
    el NIVEL: una lista de conjuntos sobre los que se mide el k marginal y se
    toma la mediana.
    """
    fac: dict = {}

    for tau in TAUS_NIVEL:                       # q01 / q99: solo nivel
        if not aplicar_nivel:
            fac[tau] = {"__marg__": 1.0}         # sin nivel no hay nada que ajustar
            continue
        fac[tau] = {"__marg__": float(np.clip(_nivel_marginal(fuentes_nivel, tau),
                                              K_MIN, K_MAX))}

    for tau in TAUS_FORMA:                       # q05 / q95: nivel x forma
        s_f, _, niv = _score(val_forma, tau)
        k_marg_f = _k_conformal(s_f, niv)
        k_marg_n = _nivel_marginal(fuentes_nivel, tau) if aplicar_nivel else 1.0
        tabla: dict = {}
        for pos in ORDEN:
            m  = (val_forma["pos"] == pos).to_numpy()
            nf = val_forma.loc[m, "fecha_th"].nunique()
            if nf < MIN_FECHAS or not np.isfinite(k_marg_f) or k_marg_f <= 0:
                tabla[pos] = float(np.clip(k_marg_n, K_MIN, K_MAX))
                log.append(f"    q{int(tau*100):02d} {pos}: {nf} fechas "
                           f"(<{MIN_FECHAS}) — sin forma, solo nivel")
                continue
            r = _k_conformal(s_f[m], niv) / k_marg_f
            r = 1.0 + LAMBDA_ENC * (r - 1.0)     # encogimiento hacia 1
            tabla[pos] = float(np.clip(k_marg_n * r, K_MIN, K_MAX))
        fac[tau] = tabla
    return fac


def aplicar(d: pd.DataFrame, fac: dict) -> pd.DataFrame:
    """Aplica los factores y reordena. q40/q50/q60 quedan intactos."""
    out = d.copy()
    for tau, tabla in fac.items():
        col = f"q{int(tau * 100):02d}"
        _, w, _ = _score(d, tau)
        if "__marg__" in tabla:
            k = np.full(len(d), tabla["__marg__"])
        else:
            k = d["pos"].map(tabla).astype(float).to_numpy()
            k = np.where(np.isfinite(k), k, 1.0)   # sin categoría → sin cambio
        out[col] = (d["q50"].to_numpy(float)
                    + (k * w if tau > 0.5 else -k * w))
    # Reordenamiento de Chernozhukov: factores distintos por tau pueden romper
    # la monotonía ENTRE cuantiles aunque no con la mediana.
    cols = [f"q{int(t*100):02d}" for t in (0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99)]
    out[cols] = np.sort(out[cols].to_numpy(float), axis=1)
    return out


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _cobertura(d: pd.DataFrame, lo="q05", hi="q95") -> float:
    return float(((d["target"] >= d[lo]) & (d["target"] <= d[hi])).mean())


def _winkler(d: pd.DataFrame, alpha=ALPHA_90) -> float:
    L, U, y = d["q05"].to_numpy(), d["q95"].to_numpy(), d["target"].to_numpy()
    pen = np.maximum(0, L - y) + np.maximum(0, y - U)
    return float(np.mean((U - L) + (2.0 / alpha) * pen))


def _resumen(d: pd.DataFrame, etiqueta: str) -> dict:
    return {
        "version"   : etiqueta,
        "cob90"     : _cobertura(d),
        "cob98"     : _cobertura(d, "q01", "q99"),
        "ancho_MM"  : float((d["q95"] - d["q05"]).mean() / 1e6),
        "winkler_MM": _winkler(d) / 1e6,
    }


def _tabla_pos(antes: pd.DataFrame, despues: pd.DataFrame) -> pd.DataFrame:
    filas = {}
    for pos in ORDEN:
        a, b = antes[antes["pos"] == pos], despues[despues["pos"] == pos]
        if len(a) == 0:
            continue
        filas[pos] = {
            "n"          : len(a),
            "cob_antes"  : _cobertura(a),
            "cob_despues": _cobertura(b),
            "ancho_antes": (a["q95"] - a["q05"]).mean() / 1e6,
            "ancho_desp" : (b["q95"] - b["q05"]).mean() / 1e6,
        }
    t = pd.DataFrame(filas).T
    t["delta_pp"] = (t["cob_despues"] - t["cob_antes"]) * 100
    return t


# ---------------------------------------------------------------------------

def _fuentes_nivel(val: pd.DataFrame, test: pd.DataFrame, f: int,
                   fuente: str) -> tuple[list[pd.DataFrame], str]:
    """
    Conjuntos sobre los que se mide el nivel marginal del fold f.

    'val_propio'   el VAL del propio fold. Es lo que se hizo primero, y ahí
                   está el sesgo: en VAL el modelo SOBRE-cubre (k de 0.89,
                   0.70, 0.60 y 0.43 por fold) mientras en TEST sub-cubre
                   (1.43, 0.98, 1.12, 1.34). Se aplicó 0.43 donde hacía falta
                   1.34, así que ninguna forma podía salvar el resultado.
    'test_previos' el TEST de los folds anteriores, uno por fuente, y se toma
                   la mediana. Mide directamente lo que interesa —cuánto ancho
                   faltó fuera de muestra— en vez de inferirlo de VAL. Es
                   causal: el TEST del fold j termina antes de que empiece el
                   del fold f, y en producción esos resultados ya se conocen.
                   Para el fold 4 daría mediana(1.43, 0.98, 1.12) = 1.12
                   contra el 1.34 real: sub-corrige, pero es 2.6 veces mejor
                   que 0.43. Sub-corrige porque la razón TEST/VAL deriva
                   (1.60, 1.41, 1.88, 3.15) y la mediana de los previos no
                   alcanza a la tendencia.
    """
    if fuente == "val_propio":
        sub = val[val["fold"] == f]
        return ([sub] if len(sub) else []), f"VAL propio ({sub['fecha_th'].nunique()} fechas)"

    # Corte por FECHA DE RESULTADO, no por número de fold. Las ventanas de TEST
    # se solapan en fecha_th —cada una se extiende hasta 75 días hábiles más
    # allá de su último origen, invadiendo el período del fold siguiente—, así
    # que "el TEST del fold anterior" incluye resultados que todavía no habrían
    # ocurrido. Lo que corresponde es todo lo cuyo resultado ya se conoce antes
    # de empezar a proyectar este período.
    corte = test.loc[test["fold"] == f, "fecha_th"].min()
    prev  = test[test["fecha_th"] < corte]
    grupos = [g for _, g in prev.groupby("fold") if len(g)]
    return grupos, (f"TEST con resultado previo a {corte:%Y-%m-%d} "
                    f"({prev['fecha_th'].nunique()} fechas, "
                    f"{len(grupos)} fold[s])")


def _conjunto_forma(val: pd.DataFrame, test: pd.DataFrame, f: int,
                    modo: str) -> tuple[pd.DataFrame, str]:
    """
    Conjunto sobre el que se estima la FORMA r(pos) para el fold f.

    'causal_val'   VAL con resultado conocido antes de empezar el período.
                   Es el primer diseño, y hereda el mismo sesgo que tenía el
                   nivel: VAL es un período sistemáticamente más benigno que
                   TEST, así que la forma sale corta. Se vio en que 'inicio +2'
                   y 'cierre -3' no se movieron pese a recibir ensanchamiento.
    'causal_todo'  VAL Y TEST con resultado ya conocido. Mismo corte temporal,
                   pero la mitad de los datos viene de un régimen realista. Y
                   resuelve el tamaño muestral: solo TEST previo daría ~17
                   fechas por categoría de borde, por debajo del mínimo de 19;
                   la unión llega a ~43.
    'lofo'         todos los VAL menos el propio. USA DATOS POSTERIORES: no es
                   desplegable, sirve de cota superior.

    Que el nivel y la forma compartan filas no genera doble conteo: r es un
    cociente contra el marginal del mismo conjunto, así que el nivel se cancela
    dentro de la forma.
    """
    if modo == "lofo":
        sub = val[val["fold"] != f]
        return sub, f"VAL de {sub['fold'].nunique()} fold(s), {sub['fecha_th'].nunique()} fechas"
    corte = test.loc[test["fold"] == f, "fecha_th"].min()
    v = val[val["fecha_th"] < corte]
    if modo == "causal_val":
        return v, f"VAL previo a {corte:%Y-%m-%d}, {v['fecha_th'].nunique()} fechas"
    t = test[test["fecha_th"] < corte]
    u = pd.concat([v, t], ignore_index=True)
    return u, (f"VAL+TEST previos a {corte:%Y-%m-%d}, "
               f"{u['fecha_th'].nunique()} fechas "
               f"({v['fecha_th'].nunique()} de VAL + {t['fecha_th'].nunique()} de TEST)")


def _corrida(val: pd.DataFrame, test: pd.DataFrame, modo: str,
             aplicar_nivel: bool = True,
             fuente_nivel: str = "val_propio") -> pd.DataFrame:
    """Calibra cada fold con el conjunto que permita `modo` y devuelve TEST calibrado."""
    etiq_niv = f"nivel={fuente_nivel}" if aplicar_nivel else "SIN NIVEL"
    print("\n" + "=" * 78)
    print(f"FORMA '{modo}' · {etiq_niv}  —  "
          + ("USA DATOS POSTERIORES, no desplegable" if modo == "lofo"
             else "solo resultados ya conocidos (desplegable)"))
    print("=" * 78)
    folds = sorted(test["fold"].unique())
    salida, tablas = [], {}
    for f in folds:
        cal, etiq_forma = _conjunto_forma(val, test, f, modo)
        fuentes, etiq = _fuentes_nivel(val, test, f, fuente_nivel)
        sub = test[test["fold"] == f]
        if len(cal) == 0 or (aplicar_nivel and not fuentes):
            print(f"\n  Fold {f}: sin conjunto de calibración — se deja sin calibrar")
            salida.append(sub)
            continue
        log: list = []
        fac = estimar_factores(cal, fuentes, log, aplicar_nivel)
        tablas[int(f)] = {str(k): v for k, v in fac.items()}
        _k = fac[0.05].get("__marg__") or list(fac[0.05].values())[3]
        print(f"\n  Fold {f}: forma de {etiq_forma}")
        print(f"           nivel de {etiq}")
        for l in log[:3]:
            print(l)
        if len(log) > 3:
            print(f"    (+{len(log)-3} categorías más sin forma)")
        salida.append(aplicar(sub, fac))
    return pd.concat(salida, ignore_index=True), tablas


def main() -> None:
    dir_modo = _dir_modo()
    val  = _adjuntar_fecha_th(_cargar(dir_modo, f"preds_val_fold*_{BANCO}_*.parquet"))
    test = _cargar(dir_modo, f"preds_base_{BANCO}_*.parquet")
    for d in (val, test):
        d["fecha_th"] = pd.to_datetime(d["fecha_th"])

    # Calendario único para VAL y TEST
    mapa = _mapa_posicion(np.concatenate([val["fecha_th"].to_numpy(),
                                          test["fecha_th"].to_numpy()]))
    val  = val.merge(mapa, on="fecha_th", how="left")
    test = test.merge(mapa, on="fecha_th", how="left")
    val  = val[val["pos"].notna()]

    print(f"VAL : {len(val):,} filas | folds {sorted(val['fold'].unique())}")
    print(f"TEST: {len(test):,} filas | folds {sorted(test['fold'].unique())}")
    print(f"      {int(test['pos'].isna().sum()):,} filas de TEST sin categoría "
          f"(meses truncados) — se dejan sin calibrar")

    # Guardia dura de leakage: en modo causal ninguna fecha usada para calibrar
    # puede caer dentro o después del TEST que se calibra. Se verifican las DOS
    # fuentes: el VAL que alimenta la forma y el TEST previo que alimenta el
    # nivel. La segunda es la delicada — usar TEST de otros folds es correcto
    # solo mientras sean anteriores.
    for f in sorted(test["fold"].unique()):
        sub = test[test["fold"] == f]
        corte = sub["fecha_th"].min()
        fuentes = {
            "forma (VAL con resultado previo)" : val[val["fecha_th"] < corte],
            "nivel (TEST con resultado previo)": test[test["fecha_th"] < corte],
        }
        for nombre, cal in fuentes.items():
            if len(cal) and cal["fecha_th"].max() >= sub["fecha_th"].min():
                raise AssertionError(
                    f"LEAKAGE fold {f} en {nombre}: la calibración llega a "
                    f"{cal['fecha_th'].max():%Y-%m-%d} y TEST empieza en "
                    f"{sub['fecha_th'].min():%Y-%m-%d}"
                )
    print("      [ok] guardia de leakage: ninguna fecha de calibración causal "
          "alcanza el TEST que calibra")

    resumen, todas_tablas = [_resumen(test, "SIN CALIBRAR")], {}
    for modo, aplicar_nivel, fuente in COMBINACIONES:
        _niv = f"nivel {fuente}" if aplicar_nivel else "solo forma"
        cal, tablas = _corrida(val, test, modo, aplicar_nivel, fuente)
        todas_tablas[f"{modo}_{_niv}"] = tablas
        resumen.append(_resumen(cal, f"{modo} · {_niv}"))

        print(f"\n  Cobertura 90% por posición del mes  [{modo} · {_niv}]")
        t = _tabla_pos(test, cal)
        print(t.assign(**{c: (t[c] * 100).round(1) for c in ("cob_antes", "cob_despues")})
              .round(1).to_string())

        print(f"\n  Cobertura 90% por fold  [{modo} · {_niv}]")
        for f in sorted(test["fold"].unique()):
            a = _cobertura(test[test["fold"] == f])
            b = _cobertura(cal[cal["fold"] == f])
            print(f"    Fold {f}:  {a:.1%}  →  {b:.1%}   ({(b-a)*100:+.1f} pp)")

        _resto = t.loc["resto del mes", "cob_despues"] if "resto del mes" in t.index else np.nan
        if np.isfinite(_resto) and _resto < 0.90:
            print(f"\n  [ALERTA] 'resto del mes' quedó en {_resto:.1%}, por debajo "
                  f"del objetivo: se estrechó donde hacía falta ancho.")

        if GUARDAR:
            DIR_CALIB.mkdir(parents=True, exist_ok=True)
            _sfx = fuente if aplicar_nivel else "soloforma"
            cal.to_parquet(
                DIR_CALIB / f"preds_mondrian_{modo}_{_sfx}_{BANCO}.parquet",
                index=False)

    print("\n" + "=" * 78)
    print("Resumen global")
    print("=" * 78)
    r = pd.DataFrame(resumen).set_index("version")
    r["cob90"] = (r["cob90"] * 100).round(2)
    r["cob98"] = (r["cob98"] * 100).round(2)
    print(r.round(1).to_string())
    print("\n  cob90 objetivo 90.0 | cob98 objetivo 98.0")
    print("  La diferencia causal vs lofo mide cuánto cuesta no poder usar")
    print("  datos posteriores: si lofo mejora mucho más, el método sirve pero")
    print("  faltan folds de historia para estimarlo causalmente.")

    if GUARDAR:
        with open(DIR_CALIB / f"factores_mondrian_{BANCO}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(todas_tablas, fh, indent=2, ensure_ascii=False)
        print(f"\n[OK] Guardado en {DIR_CALIB}")


if __name__ == "__main__":
    main()
