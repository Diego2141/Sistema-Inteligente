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

El NIVEL sale del VAL del propio fold (~194 fechas, bien estimado, absorbe la
deriva temporal). La FORMA se agrupa entre folds, porque con ~10 fechas por
categoría y fold no alcanza para un percentil 95.

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

Los dos modos de calibración
----------------------------
  causal  el fold f se calibra con los VAL de los folds j < f únicamente.
          Es lo desplegable y no usa ningún dato posterior al TEST evaluado.
  lofo    el fold f se calibra con todos los folds menos el suyo. USA DATOS
          POSTERIORES, así que no es desplegable: sirve como cota superior de
          lo que lograría la forma bien estimada, para separar "el método no
          sirve" de "no hay suficientes datos todavía".

Se reportan los dos. La diferencia entre ambos mide cuánto cuesta la
restricción causal.

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


def estimar_factores(val_forma: pd.DataFrame,
                     val_nivel: pd.DataFrame,
                     log: list) -> dict:
    """
    Devuelve {tau: {pos: k}} más {tau: "__marg__": k} para los que no llevan
    forma. val_forma alimenta la forma (agrupada) y val_nivel el nivel (del
    propio fold).
    """
    fac: dict = {}

    for tau in TAUS_NIVEL:                       # q01 / q99: solo nivel
        s_n, _, niv = _score(val_nivel, tau)
        fac[tau] = {"__marg__": float(np.clip(_k_conformal(s_n, niv), K_MIN, K_MAX))}

    for tau in TAUS_FORMA:                       # q05 / q95: nivel x forma
        s_f, _, niv = _score(val_forma, tau)
        s_n, _, _   = _score(val_nivel, tau)
        k_marg_f = _k_conformal(s_f, niv)
        k_marg_n = _k_conformal(s_n, niv)
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

def _corrida(val: pd.DataFrame, test: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Calibra cada fold con el conjunto que permita `modo` y devuelve TEST calibrado."""
    print("\n" + "=" * 78)
    print(f"MODO '{modo}'  —  " + ("solo folds anteriores (desplegable)" if modo == "causal"
                                   else "todos menos el propio (USA DATOS POSTERIORES)"))
    print("=" * 78)
    folds = sorted(test["fold"].unique())
    salida, tablas = [], {}
    for f in folds:
        if modo == "causal":
            cal = val[val["fold"] < f]
        else:
            cal = val[val["fold"] != f]
        propio = val[val["fold"] == f]
        sub    = test[test["fold"] == f]
        if len(cal) == 0 or len(propio) == 0:
            print(f"\n  Fold {f}: sin conjunto de calibración — se deja sin calibrar")
            salida.append(sub)
            continue
        log: list = []
        fac = estimar_factores(cal, propio, log)
        tablas[int(f)] = {str(k): v for k, v in fac.items()}
        n_forma = cal["fecha_th"].nunique()
        print(f"\n  Fold {f}: forma con {cal['fold'].nunique()} fold(s), "
              f"{n_forma} fechas | nivel con {propio['fecha_th'].nunique()} fechas")
        for l in log[:4]:
            print(l)
        if len(log) > 4:
            print(f"    (+{len(log)-4} categorías más sin forma)")
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

    # Guardia dura de leakage: en modo causal ninguna fecha de calibración
    # puede caer dentro o después del TEST que se calibra.
    for f in sorted(test["fold"].unique()):
        cal = val[val["fold"] < f]
        sub = test[test["fold"] == f]
        if len(cal) and cal["fecha_th"].max() >= sub["fecha_th"].min():
            raise AssertionError(
                f"LEAKAGE fold {f}: calibración llega a "
                f"{cal['fecha_th'].max():%Y-%m-%d} y TEST empieza en "
                f"{sub['fecha_th'].min():%Y-%m-%d}"
            )
    print("      [ok] guardia de leakage: ninguna fecha de calibración causal "
          "alcanza el TEST que calibra")

    resumen, todas_tablas = [_resumen(test, "SIN CALIBRAR")], {}
    for modo in ("causal", "lofo"):
        cal, tablas = _corrida(val, test, modo)
        todas_tablas[modo] = tablas
        resumen.append(_resumen(cal, modo))

        print(f"\n  Cobertura 90% por posición del mes  [{modo}]")
        t = _tabla_pos(test, cal)
        print(t.assign(**{c: (t[c] * 100).round(1) for c in ("cob_antes", "cob_despues")})
              .round(1).to_string())

        print(f"\n  Cobertura 90% por fold  [{modo}]")
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
            cal.to_parquet(DIR_CALIB / f"preds_mondrian_{modo}_{BANCO}.parquet",
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
