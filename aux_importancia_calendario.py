"""
¿Los modelos de cola usan los features de calendario en crudo, o solo el sin/cos?

Lee diag_features_por_h_<banco>_<fecha>.csv, que step005 guarda siempre —incluso
cuando el graficado falla— y responde la pregunta sin re-entrenar nada.

Por qué
-------
Tras activar dias_al_cierre_mes, dias_desde_cierre_mes e is_cierre_encaje, la
cobertura global subió 3.1pp pero el 70% de esa mejora vino del "resto del mes",
que ya cumplía el objetivo. El ancho creció ~9-11% de forma casi uniforme y el
ancho RELATIVO en los bordes del mes BAJÓ en las seis posiciones: el modelo se
ensanchó en todas partes en vez de condicionar. cierre -1 no se movió (42.5% →
42.4%).

Hipótesis: las dos codificaciones del mismo calendario compiten. El par sin/cos
es suave, captura la estacionalidad y consume la ganancia de la región; el
entero llega después sin ganancia marginal suficiente para abrir un split.

Este script la contrasta mirando la importancia por τ. Lo que decide:

  crudo ≈ 0 en q05/q95, sincos alto  → quitar sin/cos, dejar solo el entero
  crudo con importancia, sin efecto  → el problema es el cover (min_child_weight)
  ambos ≈ 0 en q05/q95               → los modelos de cola no usan calendario

Las tres señales miden cosas distintas y conviene leerlas juntas:
  gain  in-sample, participación del total de ganancia de los splits
  perm  out-of-sample sobre VAL, Δ pérdida relativo al permutar en bloques
  shap  out-of-sample, |contribución| media

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

EXPANDING     = False   # debe coincidir con la corrida que se quiere leer
AJUSTE_ARCTAN = True
BANCO         = "SISTEMA"
FECHA_TAG     = ""      # "" → la más reciente

# Features de calendario a seguir. El par sin/cos se consolida sumando ambos
# componentes, igual que hace step005 en sus heatmaps.
CALENDARIO_MES = ["dias_al_cierre_mes", "dias_desde_cierre_mes", "is_cierre_encaje"]
PARES_SINCOS   = ["dias_al_cierre_mes", "dias_al_cierre_trim", "dias_al_cierre_anio",
                  "mes", "dias_sem"]

SENALES = ["gain", "perm", "shap"]
TAUS_FOCO = ["q05", "q95"]      # los modelos de cola: son los que fijan el ancho


def _dir_modo() -> Path:
    sfx  = "_arctan" if AJUSTE_ARCTAN else ""
    return DIR_OUTPUT / f"fold_{'exp' if EXPANDING else 'roll'}{sfx}"


def _ultimo(dir_modo: Path) -> Path:
    cands = sorted(dir_modo.glob(f"diag_features_por_h_{BANCO}_*.csv"))
    if FECHA_TAG:
        cands = [c for c in cands if FECHA_TAG in c.name]
    if not cands:
        raise FileNotFoundError(
            f"No hay diag_features_por_h_{BANCO}_*.csv en {dir_modo}. "
            f"Requiere DIAG_FEATURES=True en la corrida."
        )
    if len(cands) > 1 and not FECHA_TAG:
        print(f"[aviso] {len(cands)} corridas: "
              f"{', '.join(c.stem.split('_')[-1] for c in cands)}. Se usa la última.")
    return cands[-1]


def _matriz_senal(df: pd.DataFrame, senal: str) -> pd.DataFrame:
    """
    (feature × tau) con la importancia media sobre folds y horizontes.

    El par sin/cos se suma en una fila '<base>_sincos'. En perm el Δ conjunto
    viaja en la columna _sin con 0 en _cos, así que la suma da el Δ correcto;
    en gain y shap la suma aproxima la importancia total del feature circular.
    """
    cols = [c for c in df.columns if c.startswith(f"{senal}_")]
    if not cols:
        return pd.DataFrame()
    sub = df[["tau"] + cols].copy()
    sub.columns = ["tau"] + [c[len(senal) + 1:] for c in cols]
    m = sub.groupby("tau").mean().T          # feature × tau

    for base in PARES_SINCOS:
        s, c = f"{base}_sin", f"{base}_cos"
        if s in m.index and c in m.index:
            m.loc[f"{base}_sincos"] = m.loc[s] + m.loc[c]
            m = m.drop(index=[s, c])
    return m


def _reporte(m: pd.DataFrame, senal: str) -> None:
    if m.empty:
        print(f"\n(sin columnas '{senal}_' en el CSV)")
        return

    taus = [t for t in ["q01", "q05", "q40", "q50", "q60", "q95", "q99", "mean"]
            if t in m.columns]
    m = m[taus]
    # Ranking por |importancia| dentro de cada τ: 1 = el feature más importante
    rank = m.abs().rank(ascending=False, method="min").astype(int)
    n_feat = len(m)

    filas = [f for f in CALENDARIO_MES + ["dias_al_cierre_mes_sincos"] if f in m.index]
    if not filas:
        print(f"\n(ninguno de los features de calendario está en el CSV de '{senal}')")
        return

    print(f"\n{'=' * 78}")
    print(f"{senal.upper()} — importancia media (sobre {n_feat} features)")
    print("=" * 78)
    print("valor, y entre paréntesis el puesto en el ranking de ese τ\n")
    anchos = f"{'feature':<30}" + "".join(f"{t:>15}" for t in taus)
    print(anchos)
    for f in filas:
        linea = f"{f:<30}"
        for t in taus:
            linea += f"{m.loc[f, t]:>9.4f}({rank.loc[f, t]:>3d})"
        print(linea)

    # Cuota conjunta del bloque de calendario del mes
    print(f"\n{'suma de los 4':<30}" +
          "".join(f"{m.loc[filas, t].sum():>15.4f}" for t in taus))

    print("\nLectura para los modelos de cola:")
    for t in TAUS_FOCO:
        if t not in m.columns:
            continue
        crudos = [f for f in CALENDARIO_MES if f in m.index]
        v_cru  = m.loc[crudos, t].abs().sum() if crudos else 0.0
        v_sc   = (abs(m.loc["dias_al_cierre_mes_sincos", t])
                  if "dias_al_cierre_mes_sincos" in m.index else 0.0)
        mejor_c = min((rank.loc[f, t] for f in crudos), default=n_feat)
        tot = v_cru + v_sc
        share = f"{v_cru / tot:.0%}" if tot > 0 else "n/a"
        print(f"  {t}: crudos={v_cru:.4f} (mejor puesto {mejor_c} de {n_feat}) | "
              f"sincos={v_sc:.4f} | el crudo se lleva {share} del bloque")


def main() -> None:
    ruta = _ultimo(_dir_modo())
    df = pd.read_csv(ruta)
    print(f"Archivo : {ruta.name}")
    print(f"Filas   : {len(df):,}  |  τ: {sorted(df['tau'].unique())}")
    print(f"Folds   : {sorted(df['fold'].unique())}")

    presentes = [f for f in CALENDARIO_MES
                 if any(c.endswith(f"_{f}") or c == f"gain_{f}" for c in df.columns)]
    print(f"\nFeatures de calendario en el CSV: {presentes}")
    if len(presentes) < len(CALENDARIO_MES):
        faltan = set(CALENDARIO_MES) - set(presentes)
        print(f"  [aviso] no aparecen: {sorted(faltan)} — ¿la corrida es anterior "
              f"a activarlos?")

    for senal in SENALES:
        _reporte(_matriz_senal(df, senal), senal)


if __name__ == "__main__":
    main()
