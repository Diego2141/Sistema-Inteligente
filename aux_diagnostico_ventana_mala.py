"""
¿Por qué 2025-06→2025-12 cubre 77% y 2025-01→2025-07 cubre 89%?

Diagnóstico de solo lectura sobre los parquets de predicciones que ya guardó
step005_walk_forward_cv_4.py. No re-entrena nada.

Contexto — lo que motiva este script
------------------------------------
Alineando las corridas rolling (20260811) y expanding (20260812) por ventana de
test en vez de por número de fold, la cobertura coincide dentro de 1pp:

    2024-06 → 2024-12    86.3%  vs  86.2%
    2025-01 → 2025-07    88.8%  vs  89.2%
    2025-06 → 2025-12    78.0%  vs  77.0%

...pese a diferir en modo de ventana, tamaño de entrenamiento (729 vs 1090
filas), longitud de VAL, hiperparámetros y s_factor (0.98 vs 0.40). Es decir:
la cobertura la fija el PERÍODO, no la configuración. Este script pregunta qué
tiene el período malo.

La pregunta decidible
---------------------
    SHOCK   → los fallos se concentran en pocas fechas de resultado (fecha_th).
              Excluir un puñado de fechas restaura la cobertura.
              Remedio: nada estructural; es un evento.

    RÉGIMEN → los fallos están repartidos y el factor de escala requerido está
              elevado durante todo el período.
              Remedio: calibración adaptativa (ACI), que ajusta el ancho en
              línea. CQR sobre VAL no puede verlo porque VAL es otro período.

Una fecha_th aparece en muchas filas (distintos fecha_t × h la proyectan), así
que un solo día extremo contamina decenas de filas. Por eso la concentración se
mide sobre fecha_th, no sobre filas.

Uso: ejecutar directamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración — espejo de step005_walk_forward_cv_4.py
# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

EXPANDING     = True    # debe coincidir con la corrida que se quiere leer
AJUSTE_ARCTAN = True
BANCO         = "SISTEMA"

# Selector de corrida dentro de la carpeta del modo. "" → la más reciente.
# Necesario porque fold_roll_arctan/ tiene dos corridas distintas:
#   "20260811" → rolling con VENTANA_VAL_AÑOS = 1
#   "20260812" → rolling con VENTANA_VAL_AÑOS = 0.5
# Sin esto se lee siempre la del 12 y la del 11 queda inaccesible.
FECHA_TAG     = ""

TOP_FECHAS  = [3, 5, 10, 20]   # cuántas fechas_th excluir en la prueba de shock
GRAFICO     = True


def _dir_modo() -> Path:
    sfx  = "_arctan" if AJUSTE_ARCTAN else ""
    base = "exp" if EXPANDING else "roll"
    return DIR_OUTPUT / f"fold_{base}{sfx}"


def _ultimo(dir_modo: Path, patron: str) -> Path:
    cands = sorted(dir_modo.glob(patron))
    if FECHA_TAG:
        cands = [c for c in cands if FECHA_TAG in c.name]
    if not cands:
        raise FileNotFoundError(
            f"No hay {patron} con FECHA_TAG={FECHA_TAG!r} en {dir_modo}. "
            f"Revise EXPANDING/AJUSTE_ARCTAN: deben coincidir con la corrida."
        )
    if len(cands) > 1 and not FECHA_TAG:
        print(f"[aviso] {len(cands)} corridas en {dir_modo.name}: "
              f"{', '.join(c.stem.split('_')[-1] for c in cands)}. "
              f"Se usa la última; fije FECHA_TAG para elegir otra.")
    return cands[-1]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de fallo, ancho y distancias al mediano."""
    d = df.copy()
    for c in ("fecha_t", "fecha_th"):
        d[c] = pd.to_datetime(d[c])
    d["fuera_lo"] = d["target"] < d["q05"]
    d["fuera_hi"] = d["target"] > d["q95"]
    d["fuera"]    = d["fuera_lo"] | d["fuera_hi"]
    d["ancho"]    = d["q95"] - d["q05"]
    # Distancias al mediano, con piso para no dividir por cero cuando dos
    # cuantiles colapsan (pasa en horizontes con poca señal).
    d["dist_lo"]  = (d["q50"] - d["q05"]).clip(lower=1.0)
    d["dist_hi"]  = (d["q95"] - d["q50"]).clip(lower=1.0)
    d["z_lo"]     = (d["q50"] - d["target"]) / d["dist_lo"]
    d["z_hi"]     = (d["target"] - d["q50"]) / d["dist_hi"]
    d["err_abs"]  = (d["target"] - d["q50"]).abs()
    d["mes"]      = d["fecha_th"].dt.to_period("M")
    return d


def _factores_oraculo(d: pd.DataFrame) -> tuple[float, float]:
    """
    Factor multiplicativo que cada lado necesitaría para alcanzar 5% de cola.

    k_lo = cuantil 0.95 de (q50 - y)/(q50 - q05):  cuánto habría que estirar el
    lado bajo para que solo el 5% lo cruce. 1.0 = perfectamente calibrado,
    >1 = demasiado angosto, <1 = demasiado ancho.
    """
    if len(d) == 0:
        return np.nan, np.nan
    return (float(np.quantile(d["z_lo"], 0.95)),
            float(np.quantile(d["z_hi"], 0.95)))


def _etiqueta_ventana(d: pd.DataFrame) -> str:
    return f"{d['fecha_t'].min():%Y-%m-%d}..{d['fecha_t'].max():%Y-%m-%d}"


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------

def seccion_a(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("A — Cobertura por fold, dirección del fallo y factor de escala requerido")
    print("=" * 78)
    filas = []
    for f, g in d.groupby("fold"):
        k_lo, k_hi = _factores_oraculo(g)
        filas.append({
            "fold"      : f,
            "test"      : _etiqueta_ventana(g),
            "n"         : len(g),
            "cob90"     : 1 - g["fuera"].mean(),
            "%fuera_lo" : g["fuera_lo"].mean(),
            "%fuera_hi" : g["fuera_hi"].mean(),
            "k_lo"      : k_lo,
            "k_hi"      : k_hi,
        })
    t = pd.DataFrame(filas).set_index("fold")
    for c in ("cob90", "%fuera_lo", "%fuera_hi"):
        t[c] = t[c].map("{:.1%}".format)
    for c in ("k_lo", "k_hi"):
        t[c] = t[c].map("{:.2f}".format)
    print(t.to_string())
    print("\n  cob90 nominal = 90.0% | %fuera_lo y %fuera_hi nominales = 5.0% cada uno")
    print("  k_lo/k_hi: factor por el que habría que multiplicar la semi-amplitud")
    print("  para llevar esa cola a 5%. 1.00 = calibrado, 1.30 = 30% angosto.")


def seccion_b(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("B — ¿Los fallos se concentran en pocas fechas de resultado?")
    print("=" * 78)
    print("  topK = participación de las K fechas_th con más fallos sobre el total")
    print("  de fallos. ×unifK = esa participación dividida por K/N.")
    print("  OJO con la referencia: bajo fallos uniformes ×unif NO da 1.0 sino")
    print("  ~2-3, porque elegir 'las K con más fallos' es una selección sobre el")
    print("  máximo y las fechas_th no tienen el mismo número de filas. Verificado")
    print("  con simulación. La comparación válida es ENTRE folds: un fold con")
    print("  evento puntual debe mostrar ×unif claramente mayor que los folds")
    print("  bien calibrados, y sus fechas top deben fallar en ~100% de sus filas")
    print("  (sección E). Un fold con cambio de régimen muestra ×unif MENOR,")
    print("  porque los fallos se reparten por todas las fechas.\n")
    filas = []
    for f, g in d.groupby("fold"):
        n_miss = int(g["fuera"].sum())
        por_th = g.groupby("fecha_th")["fuera"].sum().sort_values(ascending=False)
        n_th   = len(por_th)
        fila = {"fold": f, "n_fechas_th": n_th, "n_fallos": n_miss,
                "fechas_th_con_fallo": int((por_th > 0).sum())}
        for k in TOP_FECHAS:
            share    = por_th.head(k).sum() / max(n_miss, 1)
            esperado = k / max(n_th, 1)
            fila[f"top{k}"]     = f"{share:.1%}"
            fila[f"×unif{k}"]   = f"{share / max(esperado, 1e-9):.1f}"
        filas.append(fila)
    print(pd.DataFrame(filas).set_index("fold").to_string())


def seccion_c(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("C — Perfil mensual: ¿el ancho siguió a la dispersión realizada?")
    print("=" * 78)
    g = d.groupby(["fold", "mes"])
    t = pd.DataFrame({
        "n"          : g.size(),
        "cob90"      : 1 - g["fuera"].mean(),
        "ancho_MM"   : g["ancho"].mean() / 1e6,
        "disp_MM"    : g["err_abs"].mean() / 1e6,
        "k_lo"       : g["z_lo"].quantile(0.95),
        "k_hi"       : g["z_hi"].quantile(0.95),
    })
    # ratio ancho/dispersión: si el modelo escalara bien, sería estable
    t["ancho/disp"] = t["ancho_MM"] / t["disp_MM"]
    t["cob90"] = t["cob90"].map("{:.1%}".format)
    for c in ("ancho_MM", "disp_MM"):
        t[c] = t[c].round(0)
    for c in ("k_lo", "k_hi", "ancho/disp"):
        t[c] = t[c].round(2)
    print(t.to_string())
    print("\n  disp_MM = |y − q50| medio. Si el ancho siguiera a la dispersión,")
    print("  la columna ancho/disp sería constante entre meses y entre folds.")


def seccion_d(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("D — Prueba de shock: ¿excluir las peores fechas restaura la cobertura?")
    print("=" * 78)
    filas = []
    for f, g in d.groupby("fold"):
        por_th = g.groupby("fecha_th")["fuera"].sum().sort_values(ascending=False)
        fila = {"fold": f, "cob90_full": f"{1 - g['fuera'].mean():.1%}"}
        for k in TOP_FECHAS:
            excl = set(por_th.head(k).index)
            sub  = g[~g["fecha_th"].isin(excl)]
            fila[f"sin_top{k}"] = f"{1 - sub['fuera'].mean():.1%}"
        filas.append(fila)
    print(pd.DataFrame(filas).set_index("fold").to_string())
    print("\n  SHOCK   → la cobertura salta a ~90% al quitar unas pocas fechas.")
    print("  RÉGIMEN → apenas se mueve: los fallos están repartidos.")


def seccion_e(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("E — Fechas_th con más fallos, por fold")
    print("=" * 78)
    for f, g in d.groupby("fold"):
        por_th = (g.groupby("fecha_th")
                  .agg(fallos=("fuera", "sum"), filas=("fuera", "size"),
                       lo=("fuera_lo", "sum"), hi=("fuera_hi", "sum"))
                  .sort_values("fallos", ascending=False)
                  .head(8))
        por_th["%filas"] = (por_th["fallos"] / por_th["filas"]).map("{:.0%}".format)
        print(f"\n  Fold {f}  ({_etiqueta_ventana(g)})")
        print(por_th.to_string())


def _posicion_en_mes(d: pd.DataFrame) -> pd.DataFrame:
    """
    Posición de cada fecha_th dentro de su mes, contada sobre los días hábiles
    efectivamente presentes en los datos.

    pos_ini = 1 en el primer día hábil del mes; pos_fin = 1 en el último.
    Los meses con menos de 15 fechas distintas se descartan: son los bordes de
    cada ventana de test, donde el conteo estaría truncado y etiquetaría mal.
    """
    fechas = pd.DataFrame({"fecha_th": sorted(d["fecha_th"].unique())})
    fechas["mes"] = fechas["fecha_th"].dt.to_period("M")
    cnt = fechas.groupby("mes")["fecha_th"].transform("size")
    fechas["pos_ini"] = fechas.groupby("mes").cumcount() + 1
    fechas["pos_fin"] = cnt - fechas["pos_ini"] + 1
    fechas.loc[cnt < 15, ["pos_ini", "pos_fin"]] = np.nan
    return fechas[["fecha_th", "pos_ini", "pos_fin"]]


def seccion_f(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("F — Cobertura según la posición de fecha_th dentro de su mes")
    print("=" * 78)
    print("  La sección E muestra que las fechas que más fallan son cierres y")
    print("  aperturas de mes, y que fallan en ~100% de sus filas con un único")
    print("  signo. Esta sección lo cuantifica sobre todas las fechas.\n")

    m = d.merge(_posicion_en_mes(d), on="fecha_th", how="left")
    m = m[m["pos_fin"].notna()]
    if len(m) == 0:
        print("  (sin meses completos para clasificar)")
        return

    def _bucket(r):
        if r["pos_ini"] <= 3:
            return f"inicio +{int(r['pos_ini'])}"
        if r["pos_fin"] <= 3:
            return f"cierre -{int(r['pos_fin'])}"
        return "resto del mes"

    m["pos"] = m.apply(_bucket, axis=1)
    orden = ["inicio +1", "inicio +2", "inicio +3", "resto del mes",
             "cierre -3", "cierre -2", "cierre -1"]

    g = m.groupby("pos")
    t = pd.DataFrame({
        "n"         : g.size(),
        "cob90"     : 1 - g["fuera"].mean(),
        "%fuera_lo" : g["fuera_lo"].mean(),
        "%fuera_hi" : g["fuera_hi"].mean(),
        "disp_MM"   : g["err_abs"].mean() / 1e6,
        "ancho_MM"  : g["ancho"].mean() / 1e6,
    }).reindex(orden)
    t["ancho/disp"] = (t["ancho_MM"] / t["disp_MM"]).round(2)
    for c in ("cob90", "%fuera_lo", "%fuera_hi"):
        t[c] = t[c].map("{:.1%}".format)
    for c in ("disp_MM", "ancho_MM"):
        t[c] = t[c].round(0)
    print(t.to_string())

    print("\n  Lo mismo por fold, solo cobertura:")
    tf = (1 - m.groupby(["fold", "pos"])["fuera"].mean()).unstack("pos").reindex(
        columns=orden)
    print(tf.apply(lambda _c: _c.map("{:.1%}".format)).to_string())

    print("\n  Si la cobertura cae en los extremos del mes y disp_MM sube")
    print("  mientras ancho_MM se queda quieto, el modelo conoce la posición")
    print("  (tiene dias_al_cierre_mes_sin/cos) pero no ensancha el intervalo.")


def grafico(d: pd.DataFrame, dir_modo: Path, fecha_tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = d.groupby(["fold", "mes"])
    cob = (1 - g["fuera"].mean()).rename("cob")
    anc = (g["ancho"].mean() / 1e6).rename("ancho")
    dis = (g["err_abs"].mean() / 1e6).rename("disp")
    t = pd.concat([cob, anc, dis], axis=1).reset_index()
    t["x"] = t["mes"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for f, sub in t.groupby("fold"):
        ax1.plot(sub["x"], sub["cob"], marker="o", lw=1.4, label=f"Fold {f}")
        ax2.plot(sub["x"], sub["ancho"], marker="o", lw=1.4, label=f"Fold {f} ancho")
        ax2.plot(sub["x"], sub["disp"] * 4, marker="s", ls="--", lw=1.0,
                 label=f"Fold {f} disp×4")
    ax1.axhline(0.90, color="#DC2626", ls="--", lw=1.2, label="objetivo 90%")
    ax1.set_ylabel("Cobertura 90%")
    ax1.set_title("Cobertura mensual y escala del intervalo vs dispersión realizada")
    ax1.legend(fontsize=7, ncol=3)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("MM USD")
    ax2.set_xlabel("Mes de fecha_th")
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    ruta = dir_modo / f"diag_ventana_mala_{BANCO}_{fecha_tag}.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"\n[OK] Gráfico: {ruta.name}")


def main() -> None:
    dir_modo = _dir_modo()
    ruta = _ultimo(dir_modo, f"preds_base_{BANCO}_*.parquet")
    df   = pd.read_parquet(ruta)
    d    = _preparar(df)

    print(f"Archivo : {ruta.name}")
    print(f"Filas   : {len(d):,}  |  folds: {sorted(d['fold'].unique())}")
    print(f"fecha_th: {d['fecha_th'].min():%Y-%m-%d} → {d['fecha_th'].max():%Y-%m-%d}")

    seccion_a(d)
    seccion_b(d)
    seccion_c(d)
    seccion_d(d)
    seccion_e(d)
    seccion_f(d)

    if GRAFICO:
        grafico(d, dir_modo, ruta.stem.split("_")[-1])


if __name__ == "__main__":
    main()
