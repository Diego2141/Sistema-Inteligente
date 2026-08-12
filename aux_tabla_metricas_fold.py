"""
Tablas por fold × grupo de horizonte a partir del parquet de métricas que ya
guardó step005_walk_forward_cv_4.py. No re-entrena nada: solo lee y agrega.

Existe porque la consola de step005 imprime el pinball (y el RMSE, el Winkler y
el CRPS) promediados sobre todos los folds, mientras que la cobertura sí se
desglosa por fold. Ese promedio esconde exactamente lo que interesa: en la
corrida del 2026-08-12 los folds fueron de 81.3% a 89.2% de cobertura, así que
una media sobre folds mezcla regímenes muy distintos.

Uso: ejecutar directamente. Toma el parquet más reciente de la carpeta del modo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración — espejo de step005_walk_forward_cv_4.py
# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

EXPANDING     = True    # debe coincidir con la corrida que se quiere leer
AJUSTE_ARCTAN = True

BANCO = "SISTEMA"

# Métricas a desglosar. Las de escala se imprimen en MM USD; las de proporción,
# en porcentaje.
METRICAS_ESCALA     = ["pinball_q50", "rmse", "winkler_90", "crps"]
METRICAS_PROPORCION = ["coverage_90", "coverage_98",
                       "val_coverage_90", "val_coverage_98"]

BINS   = [1, 5, 15, 30, 50, 75]
LABELS = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]


def _dir_modo() -> Path:
    sfx  = "_arctan" if AJUSTE_ARCTAN else ""
    base = "exp" if EXPANDING else "roll"
    return DIR_OUTPUT / f"fold_{base}{sfx}"


def _ultimo_parquet(dir_modo: Path) -> Path:
    """El parquet de métricas más reciente por fecha en el nombre."""
    cands = sorted(dir_modo.glob(f"metricas_{BANCO}_*.parquet"))
    if not cands:
        raise FileNotFoundError(
            f"No hay metricas_{BANCO}_*.parquet en {dir_modo}. "
            f"Revise EXPANDING/AJUSTE_ARCTAN: deben coincidir con la corrida."
        )
    return cands[-1]


def _tabla(df: pd.DataFrame, metrica: str) -> pd.DataFrame:
    return (
        df.groupby(["fold", "h_grupo"], observed=True)[metrica]
        .mean()
        .unstack("h_grupo")
        .rename(index=lambda f: f"Fold {f}")
    )


def main() -> None:
    dir_modo = _dir_modo()
    ruta = _ultimo_parquet(dir_modo)
    df = pd.read_parquet(ruta)
    df["h_grupo"] = pd.cut(df["h"], bins=BINS, labels=LABELS)

    print(f"Archivo : {ruta.name}")
    print(f"Filas   : {len(df):,}  ({df['fold'].nunique()} folds × "
          f"{df['h'].nunique()} horizontes)")

    for m in METRICAS_ESCALA:
        if m not in df.columns:
            continue
        t = _tabla(df, m)

        print(f"\n{'=' * 68}\n{m} por fold y grupo de horizonte (MM USD)")
        print((t / 1e6).round(1).to_string())

        # Normalizado por fold: aísla la FORMA del perfil por horizonte de la
        # escala del fold. Necesario porque el pinball está en unidades del
        # target y la serie crece ~25% cada 6 meses, así que la columna
        # absoluta mezcla "el modelo empeoró" con "los flujos son más grandes".
        print(f"\n{m} normalizado (cada fold / su media sobre h)")
        print(t.div(t.mean(axis=1), axis=0).round(3).to_string())

        # Y la lectura inversa: cada grupo relativo al fold 1, que responde
        # "¿cuánto creció la pérdida entre folds?" a horizonte comparable.
        print(f"\n{m} relativo al Fold 1 (por columna)")
        print(t.div(t.iloc[0], axis=1).round(3).to_string())

    for m in METRICAS_PROPORCION:
        if m not in df.columns:
            continue
        print(f"\n{'=' * 68}\n{m} por fold y grupo de horizonte")
        print(_tabla(df, m).applymap("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
