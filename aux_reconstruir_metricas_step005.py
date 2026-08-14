"""
Reconstruye metricas_*.parquet, preds_base_*.parquet y tablas_resumen_*.xlsx
de una corrida de step005_walk_forward_cv_4.py que llegó a entrenar todos
los folds pero se cayó en la Sección 5 (consolidación) o después, SIN
volver a entrenar nada.

Por qué funciona
-----------------
Cada fold escribe su parquet de predicciones (preds_test_foldNN_*.parquet,
preds_val_foldNN_*.parquet) apenas termina de entrenar sus 74 horizontes —
antes de llegar a la Sección 5. Esos archivos ya tienen todo lo necesario
para recalcular las métricas (columnas q01..q99, mean, target): este script
llama a la MISMA función calcular_metricas() de step005 sobre cada (fold, h),
así que los números salen idénticos a los que hubiera producido la corrida
si no se hubiera caído. Lo único que NO se puede reconstruir sin los modelos
en memoria es n_trees_* (árboles efectivos) y los diagnósticos de
importancia (gain/perm/SHAP) — todo lo demás (RMSE, pinball por cuantil y
promedio, coverage 90/98 TEST y VAL, Winkler, CRPS, calibración hit-rate,
y las tablas por fold × grupo de horizonte) sale completo.

Uso
---
Ajustar FECHA abajo (el sufijo YYYYMMDD de los archivos ya guardados) y
ejecutar. Si algún preds_val_foldNN no existe (fold sin VAL, poco común),
ese fold simplemente no aporta a las métricas de VAL — el resto sigue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import step005_walk_forward_cv_4 as s5

# ─────────────────────────────────────────────────────────────────────────────
# Ajustar antes de correr: el sufijo de fecha de los preds_test/val_foldNN_*
# ya guardados en disco (visible en el nombre de archivo, p.ej. "20260814").
# ─────────────────────────────────────────────────────────────────────────────
FECHA = "20260814"
BANCO = s5.BANCO


def _col_to_tau() -> dict[str, float]:
    return {f"q{int(t * 100):02d}": t for t in s5.TAUS}


def _preds_dict_desde_wide(df_h: pd.DataFrame, col_tau: dict[str, float]) -> dict:
    """Reconstruye {tau: array, "mean": array} desde las columnas anchas guardadas."""
    d = {tau: df_h[col].to_numpy() for col, tau in col_tau.items() if col in df_h.columns}
    if "mean" in df_h.columns:
        d["mean"] = df_h["mean"].to_numpy()
    return d


def main() -> None:
    dir_modo = s5._dir_modo()
    col_tau = _col_to_tau()

    test_paths = sorted(dir_modo.glob(f"preds_test_fold*_{BANCO}_{FECHA}.parquet"))
    if not test_paths:
        raise FileNotFoundError(
            f"No se encontró ningún preds_test_fold*_{BANCO}_{FECHA}.parquet en "
            f"{dir_modo} — revisa FECHA (debe coincidir con el sufijo de los "
            f"archivos ya guardados por la corrida caída)."
        )
    print(f"Carpeta: {dir_modo}")
    print(f"Folds encontrados (TEST): {[p.name for p in test_paths]}")

    val_by_fold: dict[int, pd.DataFrame] = {}
    for p in dir_modo.glob(f"preds_val_fold*_{BANCO}_{FECHA}.parquet"):
        dfv = pd.read_parquet(p)
        fold_n = int(dfv["fold"].iloc[0])
        val_by_fold[fold_n] = dfv
    print(f"Folds encontrados (VAL):  {sorted(val_by_fold)}")

    resultados = []
    fold_scaffolds = []

    for p in test_paths:
        df_test = pd.read_parquet(p)
        fold_n = int(df_test["fold"].iloc[0])
        fold_scaffolds.append(df_test)
        df_val = val_by_fold.get(fold_n)

        for h_val, df_h in df_test.groupby("h"):
            preds_test = _preds_dict_desde_wide(df_h, col_tau)
            y_test = df_h["target"].to_numpy()

            preds_val = y_val = None
            if df_val is not None:
                df_hv = df_val[df_val["h"] == h_val]
                if len(df_hv) > 0:
                    preds_val = _preds_dict_desde_wide(df_hv, col_tau)
                    y_val = df_hv["target"].to_numpy()

            row = s5.calcular_metricas(preds_test, y_test, int(h_val), fold_n,
                                        preds_val, y_val)
            resultados.append(row)

        print(f"  Fold {fold_n}: {df_test['h'].nunique()} horizontes recalculados")

    if not resultados:
        print("⚠  Sin filas — nada que guardar.")
        return

    # ── preds_base (equivalente a la Sección 5 que se cayó) ─────────────────
    col_order = ["banco", "fold", "fecha_t", "fecha_th", "h", "target",
                 "q01", "q05", "q40", "q50", "q60", "q95", "q99", "mean"]
    df_all = pd.concat(fold_scaffolds, ignore_index=True)
    ordered = [c for c in col_order if c in df_all.columns]
    ruta_preds = dir_modo / f"preds_base_{BANCO}_{FECHA}.parquet"
    df_all[ordered].to_parquet(ruta_preds, index=False)
    print(f"\n✓ Guardado: {ruta_preds}  ({len(df_all):,} filas)")

    # ── métricas (equivalente a la Sección 6) ────────────────────────────────
    df_res = pd.DataFrame(resultados)
    ruta_met = dir_modo / f"metricas_{BANCO}_{FECHA}.parquet"
    df_res.to_parquet(ruta_met, index=False)
    print(f"✓ Métricas: {ruta_met}  ({len(df_res):,} filas)")

    bins   = [1, 5, 15, 30, 50, 75]
    labels = ["h02-05", "h06-15", "h16-30", "h31-50", "h51-75"]
    df_res["h_grupo"] = pd.cut(df_res["h"], bins=bins, labels=labels)

    _tablas: dict = {}

    print("\nRMSE medio por grupo de horizonte:")
    print(s5._reg(_tablas, "rmse_por_grupo",
               df_res.groupby("h_grupo", observed=True)["rmse"].mean().round(0)))

    if "pinball_q50" in df_res.columns:
        print("\nPinball q50 medio por grupo de horizonte:")
        print(s5._reg(_tablas, "pinball_q50_por_grupo",
                   df_res.groupby("h_grupo", observed=True)["pinball_q50"]
                   .mean().round(4)))

        print("\nPinball q50 por fold y grupo de horizonte:")
        _pb50 = (
            df_res.groupby(["fold", "h_grupo"], observed=True)["pinball_q50"]
            .mean()
            .unstack("h_grupo")
        )
        _pb50.index = ["Fold {}".format(f) for f in _pb50.index]
        s5._reg(_tablas, "pinball_q50_fold_grupo", _pb50)
        print(_pb50.round(0).to_string())

    _pinball_cols = [c for c in df_res.columns
                      if c.startswith("pinball_q") and not c.startswith("pinball_rel")]
    if _pinball_cols:
        df_res["pinball_prom"] = df_res[_pinball_cols].mean(axis=1)

        print(f"\nPinball promedio ({len(_pinball_cols)} cuantiles) por grupo de horizonte (MM USD):")
        _pbp = df_res.groupby("h_grupo", observed=True)["pinball_prom"].mean() / 1e6
        s5._reg(_tablas, "pinball_prom_por_grupo", _pbp.round(3))
        print(_pbp.round(3))

        print("\nPinball promedio por fold y grupo de horizonte (MM USD):")
        _pbp_fg = (
            df_res.groupby(["fold", "h_grupo"], observed=True)["pinball_prom"]
            .mean()
            .unstack("h_grupo") / 1e6
        )
        _pbp_fg.index = ["Fold {}".format(f) for f in _pbp_fg.index]
        s5._reg(_tablas, "pinball_prom_fold_grupo", _pbp_fg)
        print(_pbp_fg.round(3).to_string())

    if "coverage_90" in df_res.columns:
        print("\nCobertura empírica 90% [Q05-Q95] TEST media por grupo de horizonte:")
        _c90 = df_res.groupby("h_grupo", observed=True)["coverage_90"].mean()
        s5._reg(_tablas, "cov90_test_por_grupo", _c90)
        print(_c90.map("{:.1%}".format))

        print("\nCobertura 90% TEST por fold y grupo de horizonte:")
        _t90 = (
            df_res.groupby(["fold", "h_grupo"], observed=True)["coverage_90"]
            .mean()
            .unstack("h_grupo")
        )
        _t90.index = ["Fold {}".format(f) for f in _t90.index]
        s5._reg(_tablas, "cov90_test_fold_grupo", _t90)
        print(_t90.applymap("{:.1%}".format).to_string())

    if "coverage_98" in df_res.columns:
        print("\nCobertura empírica 98% [Q01-Q99] TEST media por grupo de horizonte:")
        _c98 = df_res.groupby("h_grupo", observed=True)["coverage_98"].mean()
        s5._reg(_tablas, "cov98_test_por_grupo", _c98)
        print(_c98.map("{:.1%}".format))

        print("\nCobertura 98% TEST por fold y grupo de horizonte:")
        _t98 = (
            df_res.groupby(["fold", "h_grupo"], observed=True)["coverage_98"]
            .mean()
            .unstack("h_grupo")
        )
        _t98.index = ["Fold {}".format(f) for f in _t98.index]
        s5._reg(_tablas, "cov98_test_fold_grupo", _t98)
        print(_t98.applymap("{:.1%}".format).to_string())

    if "val_coverage_90" in df_res.columns:
        print("\nCobertura empírica 90% [Q05-Q95] VAL media por grupo de horizonte:")
        _v90 = df_res.groupby("h_grupo", observed=True)["val_coverage_90"].mean()
        s5._reg(_tablas, "cov90_val_por_grupo", _v90)
        print(_v90.map("{:.1%}".format))

    if "val_coverage_98" in df_res.columns:
        print("\nCobertura empírica 98% [Q01-Q99] VAL media por grupo de horizonte:")
        _v98 = df_res.groupby("h_grupo", observed=True)["val_coverage_98"].mean()
        s5._reg(_tablas, "cov98_val_por_grupo", _v98)
        print(_v98.map("{:.1%}".format))

    if "winkler_90" in df_res.columns:
        print("\nWinkler Score 90% medio por grupo de horizonte (MM USD):")
        print(s5._reg(_tablas, "winkler90_por_grupo",
                   (df_res.groupby("h_grupo", observed=True)["winkler_90"]
                    .mean() / 1e6).round(3)))

    if "crps" in df_res.columns:
        print("\nCRPS medio por grupo de horizonte (MM USD):")
        print(s5._reg(_tablas, "crps_por_grupo",
                   (df_res.groupby("h_grupo", observed=True)["crps"]
                    .mean() / 1e6).round(4)))

    calib_cols = sorted([c for c in df_res.columns if c.startswith("calib_q")])
    if calib_cols:
        print("\nCalibración hit-rate por cuantil (promedio global; debe ≈ tau nominal):")
        calib_mean = df_res[calib_cols].mean().round(3)
        tau_labels = {f"calib_q{int(t*100):02d}": f"Q{int(t*100):02d} (τ={t:.2f})" for t in s5.TAUS}
        calib_mean.index = [tau_labels.get(c, c) for c in calib_mean.index]
        s5._reg(_tablas, "calibracion_hitrate", calib_mean)
        print(calib_mean.to_string())

    print("\n⚠  No reconstruibles sin los modelos en memoria (no afectan las tablas de "
          "arriba): n_trees_* (árboles efectivos), hp_report (convergencia Optuna) y "
          "los diagnósticos gain/perm/SHAP.")

    s5.graficar_metricas(df_res, dir_modo, BANCO, FECHA)
    s5.guardar_tablas_excel(_tablas, dir_modo, BANCO, FECHA)


if __name__ == "__main__":
    main()
