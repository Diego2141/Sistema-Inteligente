# -*- coding: utf-8 -*-
"""
aux_verificar_estructura_paths.py — ¿los simulacion_paths_*.parquet de dos
corridas distintas tienen la MISMA estructura?

PARA QUE SIRVE
Estos parquet son el entregable de step006 hacia afuera: alguien mas los lee
como input. Antes de mandarlos conviene poder decir "las dos corridas entregan
el mismo esquema" con algo mas solido que abrir los dos y mirarlos.

Las dos corridas que se comparan por defecto salen de ramas de codigo
DISTINTAS del orquestador — CONJUNTO_BBVA de main_conjunto() (N=2, algoritmo
P1-P5) y SISTEMA de main() (N=1, via pipeline_simulacion del modulo vf7)— y de
geometrias de fold distintas (expanding vs rolling). Que coincidan no es
automatico: son dos `resultados.append({...})` en dos archivos, y nada los ata.
Por eso el chequeo vale la pena.

QUE VERIFICA
Estructura (falla si difiere):
  - las 5 columnas del contrato, en el mismo orden
  - los dtypes, por familia (fecha / entero / flotante)
  - la clave (fecha_t, ventana, tau) unica, sin filas repetidas
  - nulos solo donde el contrato los permite
  - la grilla de tau
Contenido (se reporta, NO falla): rango de fechas, grilla de ventanas, filas.
Son perillas de configuracion (VENTANAS, el periodo de test), no estructura —
dos corridas legitimas pueden diferir ahi y seguir siendo el mismo insumo.

COMO SE CORRE
Desde Spyder, sin argumentos:

    runfile("aux_verificar_estructura_paths.py")

Sin acceso a H: corre igual: cae al AUTOTEST sintetico, que construye los dos
parquet en un directorio temporal y ejercita las MISMAS funciones de
comparacion, incluidos los controles negativos.

Para comparar otros archivos, editar ARCHIVOS.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

###############################################################################
# Configuración
###############################################################################

# Los dos parquet a comparar. Se aceptan rutas sin el sufijo .parquet y con
# espacios de mas (ver resolver_ruta): salen copiadas de un explorador de
# archivos y casi nunca vienen limpias.
ARCHIVOS = [
    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\2. Output"
    r"\step006_simulacion\xgb_qt_expanding_310.5\CONJUNTO_BBVA_1_0.5"
    r"\simulacion_paths_CONJUNTO_BBVA.parquet",

    r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\2. Output"
    r"\step006_simulacion\xgb_qt_rolling_30.51"
    r"\simulacion_paths_SISTEMA.parquet",
]

# El contrato. Las dos ramas del orquestador lo arman por separado:
#   main_conjunto  -> step006_orquestador_vf_7.py, el resultados.append del
#                     bucle de origenes
#   main (N=1)     -> pipeline_simulacion de step006_simulacion_paths_vf7.py
# Si alguna de las dos cambia y la otra no, este script lo ve.
COLUMNAS_CONTRATO = ["fecha_t", "ventana", "tau", "percentil_acum",
                     "y_realizado_acum"]

# Familia de dtype esperada por columna. Se compara por FAMILIA y no por dtype
# exacto a proposito: int32 vs int64 y float32 vs float64 dependen de la
# plataforma y de la version de pyarrow, no de la corrida, y hacerlos fallar
# seria un falso positivo cada vez que alguien actualiza un paquete.
FAMILIA_ESPERADA = {
    "fecha_t":          "fecha",
    "ventana":          "entero",
    "tau":              "flotante",
    "percentil_acum":   "flotante",
    "y_realizado_acum": "flotante",
}

# y_realizado_acum SI puede venir NaN, y no es un defecto: los origenes mas
# recientes todavia no tienen 75 dias habiles de realizado por delante. El resto
# de las columnas no admite nulos.
COLUMNAS_SIN_NULOS = ["fecha_t", "ventana", "tau", "percentil_acum"]

###############################################################################
# Contador de checklist (convención del repo)
###############################################################################

_OK = 0
_FALLA = 0


def chk(nombre: str, cond: bool, detalle: str = "") -> bool:
    global _OK, _FALLA
    if cond:
        _OK += 1
        print(f"[OK] {nombre}" + (f"  — {detalle}" if detalle else ""))
    else:
        _FALLA += 1
        print(f"[FALLA] {nombre}" + (f"  — {detalle}" if detalle else ""))
    return cond


def info(texto: str) -> None:
    print(f"[INFO] {texto}")


###############################################################################
# Lectura
###############################################################################

def resolver_ruta(cadena: str) -> Path:
    """
    Normaliza una ruta copiada a mano y devuelve el archivo que existe.

    Tres arreglos, los tres por casos vistos al pegar rutas desde el explorador:
      1. espacios sobrantes alrededor del punto  ("...BBVA. parquet")
      2. el sufijo .parquet ausente
      3. el nombre cambiado: desde que los archivos de salida llevan la
         configuracion en el nombre (sufijo_config), el de una corrida vieja se
         llama simulacion_paths_SISTEMA.parquet y el de una nueva
         simulacion_paths_SISTEMA_1_0.5_condauto.parquet. Si el nombre pedido no
         esta, se busca en la carpeta por prefijo antes de darse por vencido.

    Si nada existe devuelve la candidata principal igual, para que el mensaje de
    error nombre la ruta que se buscó.
    """
    limpia = " ".join(str(cadena).split())
    limpia = limpia.replace(" .parquet", ".parquet").replace(". parquet", ".parquet")
    p = Path(limpia)
    if p.suffix.lower() != ".parquet":
        p = p.with_suffix(".parquet")
    try:
        if p.exists():
            return p
        # Mismo prefijo, otra configuracion en el nombre.
        raiz = p.stem.split("_1_")[0].split("_cond")[0]
        candidatos = sorted(p.parent.glob(f"{raiz}*.parquet"))
        if candidatos:
            info(f"'{p.name}' no existe; se usa '{candidatos[0].name}' "
                 f"(mismo prefijo en la misma carpeta)")
            return candidatos[0]
    except OSError:
        # H: sin montar: no es motivo para abortar, el llamador lo maneja.
        pass
    return p


def leer(ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def familia_dtype(serie: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "fecha"
    if pd.api.types.is_bool_dtype(serie):
        return "booleano"
    if pd.api.types.is_integer_dtype(serie):
        return "entero"
    if pd.api.types.is_float_dtype(serie):
        return "flotante"
    if pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie):
        return "texto"
    return str(serie.dtype)


###############################################################################
# Descripción y comparación — las funciones que el autotest ejercita
###############################################################################

def describir(df: pd.DataFrame) -> dict:
    """Resumen estructural de un parquet. Es lo que se compara entre archivos."""
    d = {
        "columnas":  list(df.columns),
        "familias":  {c: familia_dtype(df[c]) for c in df.columns},
        "dtypes":    {c: str(df[c].dtype) for c in df.columns},
        "filas":     len(df),
    }
    if "tau" in df.columns:
        d["taus"] = sorted(round(float(t), 6) for t in df["tau"].dropna().unique())
    if "ventana" in df.columns:
        vs = df["ventana"].dropna().astype(int)
        d["ventanas"] = (int(vs.min()), int(vs.max()), int(vs.nunique())) if len(vs) else None
    if "fecha_t" in df.columns:
        f = pd.to_datetime(df["fecha_t"]).dropna()
        d["fechas"] = (f.min(), f.max(), int(f.nunique())) if len(f) else None
    return d


def comparar_estructura(a: dict, b: dict) -> list[str]:
    """
    Diferencias ESTRUCTURALES entre dos resúmenes. Lista vacía = mismo esquema.

    Deliberadamente NO mira filas, rango de fechas ni grilla de ventanas: son
    perillas de configuracion (el periodo de test, VENTANAS), no estructura.
    """
    difs = []
    if a["columnas"] != b["columnas"]:
        falta_b = [c for c in a["columnas"] if c not in b["columnas"]]
        falta_a = [c for c in b["columnas"] if c not in a["columnas"]]
        if falta_a or falta_b:
            difs.append(f"columnas distintas: solo en A={falta_b}, solo en B={falta_a}")
        else:
            difs.append(f"mismas columnas en DISTINTO orden: {a['columnas']} vs "
                        f"{b['columnas']}")
    for c in a["columnas"]:
        if c in b["familias"] and a["familias"][c] != b["familias"][c]:
            difs.append(f"'{c}': tipo {a['familias'][c]} vs {b['familias'][c]} "
                        f"({a['dtypes'][c]} vs {b['dtypes'][c]})")
    if a.get("taus") != b.get("taus"):
        difs.append(f"grilla de tau distinta: {a.get('taus')} vs {b.get('taus')}")
    return difs


def verificar_uno(nombre: str, df: pd.DataFrame) -> None:
    """Los invariantes que un consumidor externo puede dar por sentados."""
    d = describir(df)

    chk(f"{nombre}: las {len(COLUMNAS_CONTRATO)} columnas del contrato, en orden",
        d["columnas"] == COLUMNAS_CONTRATO, str(d["columnas"]))

    malos = {c: d["familias"][c] for c in COLUMNAS_CONTRATO
             if c in d["familias"] and d["familias"][c] != FAMILIA_ESPERADA[c]}
    chk(f"{nombre}: dtypes de la familia esperada", not malos, str(malos))

    claves = [c for c in ("fecha_t", "ventana", "tau") if c in df.columns]
    n_dup = int(df.duplicated(subset=claves).sum()) if claves else -1
    chk(f"{nombre}: (fecha_t, ventana, tau) es clave unica", n_dup == 0,
        f"{n_dup} filas repetidas")

    nulos = {c: int(df[c].isna().sum()) for c in COLUMNAS_SIN_NULOS
             if c in df.columns and df[c].isna().any()}
    chk(f"{nombre}: sin nulos donde el contrato no los admite", not nulos, str(nulos))

    # Monotonia en tau: q01 <= q05 <= ... <= q99 dentro de cada (fecha_t,
    # ventana). Es el invariante que mas le importa a quien consuma esto: un
    # cruce de cuantiles convierte una banda en un intervalo invertido, y no lo
    # detecta ningun chequeo de esquema.
    if set(COLUMNAS_CONTRATO[:4]).issubset(df.columns):
        orden = df.sort_values(["fecha_t", "ventana", "tau"])
        dif = orden.groupby(["fecha_t", "ventana"], sort=False)["percentil_acum"].diff()
        n_cruces = int((dif < -1e-9).sum())
        chk(f"{nombre}: percentil_acum no decrece con tau (sin cruces de cuantiles)",
            n_cruces == 0, f"{n_cruces} cruces")

    # Contenido: se informa, no se juzga.
    info(f"{nombre}: {d['filas']:,} filas | taus={d.get('taus')}")
    if d.get("ventanas"):
        v0, v1, nv = d["ventanas"]
        info(f"{nombre}: ventanas {v0}..{v1} ({nv} distintas)")
    if d.get("fechas"):
        f0, f1, nf = d["fechas"]
        info(f"{nombre}: origenes {f0:%Y-%m-%d} .. {f1:%Y-%m-%d} ({nf} fechas)")


###############################################################################
# Autotest sintético — corre sin acceso a H:
###############################################################################

def _df_sintetico(n_fechas=3, ventanas=(2, 5, 10),
                  taus=(0.01, 0.05, 0.5, 0.95, 0.99)) -> pd.DataFrame:
    """Un parquet con la forma del contrato y percentiles monotonos en tau."""
    filas = []
    fechas = pd.bdate_range("2025-01-02", periods=n_fechas)
    for ft in fechas:
        for v in ventanas:
            for t in taus:
                filas.append({
                    "fecha_t": ft,
                    "ventana": int(v),
                    "tau": float(t),
                    # monotono en tau por construccion
                    "percentil_acum": float(-100 * v * (1 - t)),
                    "y_realizado_acum": float(-3 * v),
                })
    return pd.DataFrame(filas, columns=COLUMNAS_CONTRATO)


def autotest() -> int:
    """
    Ejercita describir() y comparar_estructura() — las MISMAS que usa el modo
    real— con controles negativos: cada mutacion tiene que ser detectada.
    """
    print("\n" + "=" * 78)
    print("AUTOTEST SINTETICO (no necesita acceso a H:)")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = _df_sintetico()
        b = _df_sintetico(n_fechas=5, ventanas=(2, 5, 10, 22))  # otro contenido
        (ra, rb) = (tmp / "simulacion_paths_A.parquet",
                    tmp / "simulacion_paths_B.parquet")
        a.to_parquet(ra, index=False)
        b.to_parquet(rb, index=False)

        da, db = describir(leer(ra)), describir(leer(rb))
        chk("autotest 1. dos parquet con el mismo esquema y DISTINTO contenido "
            "no producen diferencias estructurales",
            comparar_estructura(da, db) == [],
            f"{da['filas']} vs {db['filas']} filas")

        # ── Controles negativos: cada uno tiene que ser detectado ────────────
        c = _df_sintetico().rename(columns={"percentil_acum": "pctil"})
        difs = comparar_estructura(da, describir(c))
        chk("autotest 2. [neg] una columna renombrada se detecta",
            any("columnas distintas" in d for d in difs), str(difs[:1]))

        c = _df_sintetico()[["tau", "fecha_t", "ventana", "percentil_acum",
                             "y_realizado_acum"]]
        difs = comparar_estructura(da, describir(c))
        chk("autotest 3. [neg] el mismo juego de columnas en otro ORDEN se detecta",
            any("DISTINTO orden" in d for d in difs), str(difs[:1]))

        c = _df_sintetico()
        c["ventana"] = c["ventana"].astype(float)
        difs = comparar_estructura(da, describir(c))
        chk("autotest 4. [neg] un entero convertido a flotante se detecta",
            any("'ventana'" in d for d in difs), str(difs[:1]))

        c = _df_sintetico(taus=(0.05, 0.5, 0.95))
        difs = comparar_estructura(da, describir(c))
        chk("autotest 5. [neg] una grilla de tau distinta se detecta",
            any("grilla de tau" in d for d in difs), str(difs[:1]))

        c = _df_sintetico()
        c["ventana"] = c["ventana"].astype("int32")
        chk("autotest 6. int32 vs int64 NO se reporta (misma familia, no es "
            "una diferencia de corrida)",
            comparar_estructura(da, describir(c)) == [])

        # El chequeo de cruces de cuantiles tiene que poder fallar.
        c = _df_sintetico()
        i = c.index[(c["tau"] == 0.95)][0]
        c.loc[i, "percentil_acum"] = -1e9
        orden = c.sort_values(["fecha_t", "ventana", "tau"])
        dif = orden.groupby(["fecha_t", "ventana"], sort=False)["percentil_acum"].diff()
        chk("autotest 7. [neg] un cruce de cuantiles inyectado se detecta",
            int((dif < -1e-9).sum()) > 0)

        # resolver_ruta: los tres arreglos de ruta, sobre archivos que existen.
        chk("autotest 8. resolver_ruta arregla el espacio de mas",
            resolver_ruta(str(ra).replace(".parquet", ". parquet")) == ra)
        chk("autotest 9. resolver_ruta agrega el sufijo .parquet ausente",
            resolver_ruta(str(ra)[:-len(".parquet")]) == ra)
        (tmp / "simulacion_paths_C_1_0.5_condregimen.parquet").write_bytes(
            ra.read_bytes())
        chk("autotest 10. resolver_ruta encuentra el archivo cuando el nombre "
            "lleva la configuracion",
            resolver_ruta(str(tmp / "simulacion_paths_C.parquet")).name
            == "simulacion_paths_C_1_0.5_condregimen.parquet")

    return 0


###############################################################################
# Main
###############################################################################

def main() -> int:
    print("=" * 78)
    print("ESTRUCTURA DE LOS simulacion_paths_*.parquet")
    print("=" * 78)

    rutas = [resolver_ruta(a) for a in ARCHIVOS]
    faltan = []
    for r in rutas:
        try:
            if not r.exists():
                faltan.append(r)
        except OSError as e:
            print(f"[INFO] no se pudo inspeccionar {r} ({type(e).__name__}: {e})")
            faltan.append(r)

    if faltan:
        print("\nNo se pudo leer:")
        for r in faltan:
            print(f"  - {r}")
        print("\nSin acceso a H: no se puede comparar lo real. Se corre el "
              "autotest, que valida la LOGICA de comparacion.")
        autotest()
    else:
        dfs, descs = [], []
        for r in rutas:
            print(f"\n--- {r.name}")
            print(f"    {r.parent}")
            df = leer(r)
            dfs.append(df)
            descs.append(describir(df))
            verificar_uno(r.name, df)

        print("\n--- comparacion entre los dos")
        difs = comparar_estructura(descs[0], descs[1])
        chk("misma estructura: columnas, orden, tipos y grilla de tau",
            not difs, "; ".join(difs) if difs else "identicas")

        # Contenido: se informa y no se juzga — son perillas de config.
        if descs[0]["filas"] != descs[1]["filas"]:
            info(f"filas distintas ({descs[0]['filas']:,} vs {descs[1]['filas']:,}) "
                 f"— esperable: distinto periodo de test y distinta grilla VENTANAS")
        if descs[0].get("ventanas") != descs[1].get("ventanas"):
            info(f"grilla de ventanas distinta: {descs[0].get('ventanas')} vs "
                 f"{descs[1].get('ventanas')} — es la perilla VENTANAS de step006")

        # El autotest corre igual: es lo que prueba que los chequeos de arriba
        # PUEDEN fallar. Un [OK] de una comprobacion que nunca falla no dice nada.
        autotest()

    print("\n" + "=" * 78)
    print(f"RESULTADO: {_OK} OK / {_FALLA} FALLA")
    print("=" * 78)
    return 1 if _FALLA else 0


if __name__ == "__main__":
    sys.exit(main())
