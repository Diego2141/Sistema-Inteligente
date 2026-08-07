"""
aux_reporte_coverage.py
=======================
Excel con el detalle fila a fila del cálculo de coverage_90 y coverage_98.

Cómo se calculan en step005 (calcular_metricas):

    coverage_90  =  mean( (target >= q05) & (target <= q95) )
    coverage_98  =  mean( (target >= q01) & (target <= q99) )

Es decir, la fracción de observaciones de TEST cuyo valor real cae dentro del
intervalo, calculada por cada (fold, h) sobre sus ~120 filas. Los extremos
cuentan como dentro.

Desde el fix del rearrangement de cuantiles, preds_base y las métricas usan el
mismo diccionario de predicciones, así que los valores del parquet son
exactamente los que produjeron el número — este reporte es fiel, no una
reconstrucción aproximada. El script lo VERIFICA contra metricas_*.parquet.

Hojas que genera:
  detalle             una fila por (fold, h, fecha) con los indicadores
  resumen_fold_h      coverage por fold × h  (el nivel al que step005 lo calcula)
  resumen_fold_grupo  coverage por fold × h_grupo  (lo que imprime en consola)
  violaciones         solo las filas fuera del intervalo 90%, para inspección
  verificacion        contraste contra metricas_*.parquet

Uso:  python aux_reporte_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_WFCV     = BASE_SISTEMA / "2. Output" / "step005_wfcv_v4_direct"

BANCO = "SISTEMA"
MODO  = "roll_arctan"        # carpeta fold_<MODO>; ajustar según la corrida

# Mismos bins que usa step005 al imprimir las tablas por grupo
BINS   = [1, 5, 15, 30, 50, 75]
LABELS = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]

# Excel: nº máximo de filas que se vuelca en la hoja de detalle. Si el parquet
# es mayor, se escribe también un CSV con todo (el límite de Excel es ~1.05M,
# pero archivos de cientos de miles de filas se vuelven inmanejables).
MAX_FILAS_DETALLE = 200_000


def _titulo(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def _fmt_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formatea un DataFrame numérico como porcentajes.

    Se evita DataFrame.applymap / DataFrame.map a propósito: applymap se
    eliminó en pandas 2.1 y DataFrame.map no existe antes de esa versión, así
    que ninguna de las dos funciona en ambos entornos. Series.map sí.
    """
    return df.apply(lambda col: col.map(
        lambda v: f"{v:.1%}" if pd.notna(v) else ""))


def main() -> None:
    dir_fold = DIR_WFCV / f"fold_{MODO}"
    paths = sorted(dir_fold.glob(f"preds_base_{BANCO}_*.parquet"))
    if not paths:
        print(f"[ERROR] sin preds_base en {dir_fold}")
        print("        revisa MODO o corre step005 primero")
        return

    p = paths[-1]
    print(f"Leyendo {p.name}")
    df = pd.read_parquet(p)
    print(f"  {len(df):,} filas | folds {sorted(df['fold'].unique())} | "
          f"h {df['h'].min()}–{df['h'].max()}")

    faltan = [c for c in ("target", "q01", "q05", "q95", "q99") if c not in df.columns]
    if faltan:
        print(f"[ERROR] faltan columnas: {faltan}")
        return

    # ── Indicadores, exactamente como en calcular_metricas ────────────────
    df = df.dropna(subset=["target", "q01", "q05", "q95", "q99"]).copy()
    df["dentro_90"] = (df["target"] >= df["q05"]) & (df["target"] <= df["q95"])
    df["dentro_98"] = (df["target"] >= df["q01"]) & (df["target"] <= df["q99"])

    # Diagnóstico del lado de la violación: informa si el intervalo se queda
    # corto por abajo o por arriba, que es lo que distingue un sesgo de nivel
    # de un intervalo demasiado estrecho.
    df["viola_90"] = np.where(df["target"] < df["q05"], "bajo",
                       np.where(df["target"] > df["q95"], "alto", ""))
    df["viola_98"] = np.where(df["target"] < df["q01"], "bajo",
                       np.where(df["target"] > df["q99"], "alto", ""))

    # Distancia al borde violado, en MM (0 si está dentro)
    df["exceso_90_mm"] = np.where(
        df["target"] < df["q05"], (df["q05"] - df["target"]) / 1e6,
        np.where(df["target"] > df["q95"], (df["target"] - df["q95"]) / 1e6, 0.0))
    df["ancho_90_mm"] = (df["q95"] - df["q05"]) / 1e6

    df["h_grupo"] = pd.cut(df["h"], bins=BINS, labels=LABELS)

    # ── Resúmenes ─────────────────────────────────────────────────────────
    res_fh = (df.groupby(["fold", "h"], observed=True)
                .agg(n=("dentro_90", "size"),
                     n_dentro_90=("dentro_90", "sum"),
                     coverage_90=("dentro_90", "mean"),
                     n_dentro_98=("dentro_98", "sum"),
                     coverage_98=("dentro_98", "mean"),
                     ancho_90_mm=("ancho_90_mm", "mean"))
                .reset_index())

    res_fg = (df.groupby(["fold", "h_grupo"], observed=True)
                .agg(n=("dentro_90", "size"),
                     coverage_90=("dentro_90", "mean"),
                     coverage_98=("dentro_98", "mean"))
                .reset_index())

    piv_90 = res_fg.pivot(index="fold", columns="h_grupo", values="coverage_90")
    piv_98 = res_fg.pivot(index="fold", columns="h_grupo", values="coverage_98")

    _titulo("Cobertura 90% por fold y grupo (debe coincidir con la consola de step005)")
    print(_fmt_pct(piv_90).to_string())
    _titulo("Cobertura 98% por fold y grupo")
    print(_fmt_pct(piv_98).to_string())

    # ── Verificación contra metricas_*.parquet ────────────────────────────
    _titulo("Verificación: recálculo vs. lo que guardó step005")
    verif = pd.DataFrame()
    mpaths = sorted(dir_fold.glob(f"metricas_{BANCO}_*.parquet"))
    if not mpaths:
        print("  [AVISO] sin metricas_*.parquet — no se puede verificar")
    else:
        met = pd.read_parquet(mpaths[-1])
        cols = [c for c in ("fold", "h", "coverage_90", "coverage_98") if c in met.columns]
        verif = res_fh[["fold", "h", "coverage_90", "coverage_98"]].merge(
            met[cols], on=["fold", "h"], suffixes=("_recalc", "_step005"))
        for niv in ("90", "98"):
            a, b = f"coverage_{niv}_recalc", f"coverage_{niv}_step005"
            if a in verif.columns and b in verif.columns:
                d = (verif[a] - verif[b]).abs()
                verif[f"dif_{niv}"] = d
                print(f"  coverage_{niv}: max |diferencia| = {d.max():.2e}  "
                      f"| filas con dif > 1e-9: {(d > 1e-9).sum()}")
        print("  (una diferencia de 0 confirma que el parquet contiene "
              "exactamente los valores usados)")

    # ── Excel ─────────────────────────────────────────────────────────────
    _titulo("Escritura del Excel")
    cols_det = ["banco", "fold", "h", "h_grupo", "fecha_t", "fecha_th", "target",
                "q01", "q05", "q95", "q99", "dentro_90", "dentro_98",
                "viola_90", "viola_98", "exceso_90_mm", "ancho_90_mm"]
    cols_det = [c for c in cols_det if c in df.columns]
    det = df[cols_det].sort_values(["fold", "h", "fecha_t"])

    fecha = p.stem.split("_")[-1]
    ruta = dir_fold / f"reporte_coverage_{BANCO}_{fecha}.xlsx"

    if len(det) > MAX_FILAS_DETALLE:
        ruta_csv = dir_fold / f"reporte_coverage_detalle_{BANCO}_{fecha}.csv"
        det.to_csv(ruta_csv, index=False)
        print(f"  detalle completo ({len(det):,} filas) → {ruta_csv.name}")
        det_xl = det.head(MAX_FILAS_DETALLE)
    else:
        det_xl = det

    viol = det[~det["dentro_90"]].copy()

    try:
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            det_xl.to_excel(xl, sheet_name="detalle", index=False)
            res_fh.to_excel(xl, sheet_name="resumen_fold_h", index=False)
            res_fg.to_excel(xl, sheet_name="resumen_fold_grupo", index=False)
            piv_90.to_excel(xl, sheet_name="pivot_cov90")
            piv_98.to_excel(xl, sheet_name="pivot_cov98")
            viol.to_excel(xl, sheet_name="violaciones_90", index=False)
            if not verif.empty:
                verif.to_excel(xl, sheet_name="verificacion", index=False)
    except Exception as e:
        print(f"  [AVISO] no se pudo escribir el Excel ({e}) — se guardan CSVs")
        for nom, d in (("detalle", det), ("resumen_fold_h", res_fh),
                       ("resumen_fold_grupo", res_fg), ("violaciones_90", viol)):
            d.to_csv(dir_fold / f"reporte_coverage_{nom}_{BANCO}_{fecha}.csv",
                     index=False)
        return

    print(f"  {ruta.name}")
    print(f"    detalle            : {len(det_xl):,} filas")
    print(f"    resumen_fold_h     : {len(res_fh):,} filas (fold × h)")
    print(f"    resumen_fold_grupo : {len(res_fg):,} filas")
    print(f"    violaciones_90     : {len(viol):,} filas "
          f"({len(viol)/len(det):.1%} del total)")

    # ── Lectura rápida del lado de las violaciones ────────────────────────
    _titulo("Dónde falla el intervalo 90%")
    lado = det.loc[~det["dentro_90"], "viola_90"].value_counts()
    tot = len(det)
    for k in ("bajo", "alto"):
        n = int(lado.get(k, 0))
        print(f"   target por {k:>4} del intervalo: {n:>6,}  ({n/tot:.1%} del total)")
    print(f"\n   esperado si el modelo estuviera bien calibrado: 5.0% por cada lado")
    print("   un desbalance entre 'bajo' y 'alto' indica sesgo de nivel;")
    print("   ambos por encima de 5% indica intervalo demasiado estrecho")


if __name__ == "__main__":
    main()
