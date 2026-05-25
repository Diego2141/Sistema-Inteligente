# -*- coding: utf-8 -*-
"""
aux_fanchart_horizontes.py
Fan chart del modelo desde la última fecha del TEST con datos realizados.

Lógica:
  - Se para en la última fecha t del TEST que tiene al menos h=1 realizado.
  - Muestra las bandas del modelo Q01-Q99 y Q05-Q95 para h=1..90.
  - Superpone la ruta realizada D-R hasta donde hay datos disponibles;
    para h mayores la línea se corta y las bandas continúan solas
    (proyección pura hacia el futuro).

Esto replica exactamente el uso operativo: parado en t, el operador ve
la proyección del modelo para los próximos 90 días hábiles.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_SISTEMA = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
RUTA_MATRIZ  = BASE_SISTEMA / "1. Data"   / "Clean"  / "matriz_features.parquet"
DIR_MODELOS  = BASE_SISTEMA / "2. Output" / "modelos"
DIR_OUTPUT   = BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO        = "SISTEMA"
SEMANAS_VAL  = 26
SEMANAS_TEST = 18   # = h_max/5 → realizado completo h=1..90 en fan chart


# ── 1. Cargar modelo más reciente ────────────────────────────────────────────
def cargar_modelos(banco: str, dir_modelos: Path) -> tuple[dict, list[str]]:
    metas = sorted(dir_modelos.glob(f"metadata_{banco}_*.json"), reverse=True)
    if not metas:
        raise FileNotFoundError(f"No se encontró metadata para banco={banco}")
    meta  = json.loads(metas[0].read_text(encoding="utf-8"))
    fecha = metas[0].stem.split("_")[-1]
    cols_feat = meta["features"]
    quantiles = meta["quantiles"]

    modelos = {}
    for tau in quantiles:
        ruta = dir_modelos / f"lgbm_{banco}_q{int(tau*100):02d}_{fecha}.txt"
        if not ruta.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        modelos[tau] = lgb.Booster(model_file=str(ruta))

    print(f"Modelo cargado : {metas[0].name}")
    print(f"  Quantiles    : {quantiles}")
    print(f"  Features     : {len(cols_feat)}")
    return modelos, cols_feat


# ── 2. Leer datos, split y seleccionar fecha de origen ───────────────────────
def leer_y_preparar(banco: str, cols_feat: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_parquet(RUTA_MATRIZ, filters=[("banco", "==", banco)])
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

    fechas = np.sort(df["fecha_t"].unique())
    n = len(fechas)
    n_test = min(SEMANAS_TEST * 5, n // 6)
    n_val  = min(SEMANAS_VAL  * 5, n // 4)
    corte_val  = fechas[n - n_test - n_val]
    corte_test = fechas[n - n_test]

    df_train = df[df["fecha_t"] <  corte_val].copy()
    df_test  = df[df["fecha_t"] >= corte_test].copy()

    # Imputar con mediana de train (sin leak)
    cols_excluir = {"fecha_t", "banco", "target"}
    cols_num = [c for c in df.columns if c not in cols_excluir]
    medianas = df_train[cols_num].median()
    df_test[cols_num] = df_test[cols_num].fillna(medianas)

    print(f"\nTEST completo : {df_test['fecha_t'].min().date()} → "
          f"{df_test['fecha_t'].max().date()} ({df_test['fecha_t'].nunique()} fechas)")

    # Primera fecha del TEST como origen: los ~65 días hábiles del período
    # de test corresponden a h=1..65 con realizado disponible para comparar.
    # h=66..90 queda como proyección pura sin datos aún.
    fecha_origen = pd.Timestamp(corte_test)
    print(f"Fecha de origen: {fecha_origen.date()}  "
          f"(primera fecha del TEST — permite comparar h=1..{n_test} con realizado)")

    # Filtrar las filas de la fecha de origen — puede estar en df_test o df completo
    # porque corte_test es exactamente la primera fecha del TEST
    df_fecha = df[df["fecha_t"] == fecha_origen].copy().sort_values("h")
    cols_num_ok = [c for c in cols_num if c in df_fecha.columns]
    df_fecha[cols_num_ok] = df_fecha[cols_num_ok].fillna(medianas)

    # Alinear cols_feat con las disponibles
    cols_ok = [c for c in cols_feat if c in df_fecha.columns]
    faltantes = set(cols_feat) - set(cols_ok)
    if faltantes:
        print(f"  Advertencia: {len(faltantes)} features del modelo ausentes en el parquet — se imputan con 0")
        for c in faltantes:
            df_fecha[c] = 0.0

    return df_fecha, fecha_origen


# ── 3. Predecir para todos los h desde la fecha de origen ────────────────────
def predecir(modelos: dict, df_fecha: pd.DataFrame, cols_feat: list[str]) -> pd.DataFrame:
    cols_ok = [c for c in cols_feat if c in df_fecha.columns]
    X = df_fecha[cols_ok].copy()

    resultado = df_fecha[["h", "target"]].copy().reset_index(drop=True)
    for tau, model in modelos.items():
        resultado[f"q{int(tau*100):02d}"] = model.predict(X)

    # Corrección de monotonicidad fila a fila
    q_cols = sorted([c for c in resultado.columns if c.startswith("q")])
    resultado[q_cols] = np.sort(resultado[q_cols].values, axis=1)
    return resultado


# ── 4. Graficar ───────────────────────────────────────────────────────────────
def _dibujar_bandas(ax, hs, resultado):
    """Dibuja bandas Q01-Q99, Q05-Q95 y mediana Q50. Reutilizable en ambos subplots."""
    ax.fill_between(hs,
                    resultado["q01"] / 1e6, resultado["q99"] / 1e6,
                    alpha=0.12, color="steelblue", label="Q01–Q99 (98%)")
    ax.fill_between(hs,
                    resultado["q05"] / 1e6, resultado["q95"] / 1e6,
                    alpha=0.28, color="steelblue", label="Q05–Q95 (90%)")
    # Bordes explícitos de Q05 y Q95 para verificar visualmente que Q50 queda entre ellos
    ax.plot(hs, resultado["q05"] / 1e6,
            color="steelblue", lw=1.0, ls=":", alpha=0.7, label="Q05 / Q95 (borde)")
    ax.plot(hs, resultado["q95"] / 1e6,
            color="steelblue", lw=1.0, ls=":", alpha=0.7)
    ax.plot(hs, resultado["q50"] / 1e6,
            color="crimson", lw=2.0, label="Mediana predicha (Q50)", zorder=5)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlim(hs.min() - 1, hs.max() + 1)


def graficar(resultado: pd.DataFrame, fecha_origen: pd.Timestamp, banco: str):
    hs        = resultado["h"].values
    realizado = resultado["target"].values / 1e6
    mask_real = ~np.isnan(realizado)
    h_max_real = int(resultado.loc[mask_real, "h"].max()) if mask_real.any() else 0

    titulo_base = (
        f"Fan Chart — {banco}  |  Fecha de origen: {fecha_origen.strftime('%d %b %Y')}  |  "
        f"h = 1 … 90 días hábiles\n"
        f"Realizado disponible: h = 1 … {h_max_real}  |  "
        f"Proyección pura: h = {h_max_real + 1} … 90"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                   gridspec_kw={"hspace": 0.08})

    # ── Subplot superior: bandas + realizado ─────────────────────────────
    _dibujar_bandas(ax1, hs, resultado)

    if mask_real.any():
        ax1.plot(hs[mask_real], realizado[mask_real],
                 color="black", lw=2, label="Realizado (D−R)", zorder=5)
        ax1.scatter(hs[mask_real], realizado[mask_real],
                    color="black", s=18, zorder=6)

    if h_max_real > 0:
        ax1.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")

    ax1.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax1.set_title(titulo_base, fontweight="bold", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ── Subplot inferior: solo bandas del modelo ─────────────────────────
    _dibujar_bandas(ax2, hs, resultado)

    if h_max_real > 0:
        ax2.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")

    ax2.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)
    ax2.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax2.set_title("Solo proyección del modelo (sin realizado) — mediana visible",
                  fontsize=10, style="italic")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    nombre = f"fanchart_{banco}_{fecha_origen.strftime('%Y%m%d')}_h1_h90.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGuardado: {DIR_OUTPUT / nombre}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modelos, cols_feat     = cargar_modelos(BANCO, DIR_MODELOS)
    df_fecha, fecha_origen = leer_y_preparar(BANCO, cols_feat)
    resultado              = predecir(modelos, df_fecha, cols_feat)

    # ── Verificación de ordenamiento: q01 ≤ q05 ≤ q50 ≤ q95 ≤ q99 ──────
    q_cols = ["q01", "q05", "q50", "q95", "q99"]
    violaciones = (
        (resultado["q05"] < resultado["q01"]).sum() +
        (resultado["q50"] < resultado["q05"]).sum() +
        (resultado["q95"] < resultado["q50"]).sum() +
        (resultado["q99"] < resultado["q95"]).sum()
    )
    print(f"\nVerificación de monotonicidad: {violaciones} violaciones de q01≤q05≤q50≤q95≤q99")
    print("\nMuestra de quantiles en h clave (MM USD):")
    h_muestra = [1, 5, 10, 20, 30, 50, 61, 70, 90]
    sub = resultado[resultado["h"].isin(h_muestra)][["h"] + q_cols].copy()
    for c in q_cols:
        sub[c] = (sub[c] / 1e6).round(0)
    print(sub.to_string(index=False))

    graficar(resultado, fecha_origen, BANCO)
