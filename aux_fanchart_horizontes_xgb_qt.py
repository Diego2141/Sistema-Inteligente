# -*- coding: utf-8 -*-
"""
aux_fanchart_horizontes_xgb_qt.py
Fan chart del modelo XGBoost QT — lee predicciones de step005 (parquet).

Por cada fecha de origen seleccionada genera un PNG con 3 subplots:
    1. Flujo diario D-R: bandas + realizado
    2. Flujo diario D-R: solo modelo (mediana visible)
    3. Flujo neto acumulado D-R: bandas + realizado acumulado

Requiere haber corrido step005_walk_forward_cv_3.py para generar los parquets:
    preds_base_{banco}_{fecha}.parquet    <- predicciones modelo puro
    preds_overlay_{banco}_{fecha}.parquet <- con overlay sobreencaje

Parámetros ajustables:
  PASO_FECHAS   : cada cuántos días hábiles tomar una fecha de origen (2 = bisemanal)
  N_FECHAS_MAX  : límite de gráficos a generar (None = todos)
  MOSTRAR_OVERLAY: False = base; True = con overlay sobreencaje
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday,
    USFederalHolidayCalendar, Easter,
)
from pandas.tseries.offsets import CustomBusinessDay, Day as _Day

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_SISTEMA      = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_PREDS_STEP005 = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"
DIR_OUTPUT        = BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes_xgb_qt"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO = "SISTEMA"

# ── Exportación Excel ─────────────────────────────────────────────────────────
EXPORTAR_EXCEL = True   # True = genera Excel al final del script; False = solo PNGs
RUTA_EXCEL = (BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes_xgb_qt"
              / "fanchart_datos.xlsx")

# ── Parámetros del loop ───────────────────────────────────────────────────────
PASO_FECHAS    = 2     # cada 2 días hábiles; usar 1 para todas las fechas
N_FECHAS_MAX   = None  # None = generar todos; ej: 6 para las 6 primeras fechas válidas
MOSTRAR_OVERLAY = False  # False = predicciones base; True = con overlay sobreencaje

COLOR_BANDA = "tomato"


# ── 1. Cargar parquet de step005 ──────────────────────────────────────────────
def cargar_parquet(banco: str) -> pd.DataFrame:
    tipo = "overlay" if MOSTRAR_OVERLAY else "base"
    candidatos = sorted(DIR_PREDS_STEP005.glob(f"*/preds_{tipo}_{banco}_*.parquet"))
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró preds_{tipo}_{banco}_*.parquet en {DIR_PREDS_STEP005}\n"
            f"  -> Corre step005_walk_forward_cv_3.py primero"
        )
    ruta = candidatos[-1]
    df = pd.read_parquet(ruta)
    df["fecha_t"] = pd.to_datetime(df["fecha_t"])
    print(f"[OK] Parquet '{tipo}' cargado: {ruta.name}  ({len(df):,} filas)")
    return df


# ── 2. Obtener fechas válidas ─────────────────────────────────────────────────
def fechas_validas(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["fecha_t"].unique())


# ── 3. Preparar resultado para una fecha ─────────────────────────────────────
def preparar_resultado(df: pd.DataFrame, fecha_origen: pd.Timestamp) -> pd.DataFrame:
    res = df[df["fecha_t"] == fecha_origen].sort_values("h").reset_index(drop=True)
    if res.empty:
        raise ValueError(f"Sin datos en parquet para fecha_t={fecha_origen.date()}")
    return res


# ── 4. Graficar ───────────────────────────────────────────────────────────────
def _dibujar_bandas(ax, hs, resultado):
    ax.fill_between(hs,
                    resultado["q01"] / 1e6, resultado["q99"] / 1e6,
                    alpha=0.12, color=COLOR_BANDA, label="Q01–Q99 (98%)")
    ax.fill_between(hs,
                    resultado["q05"] / 1e6, resultado["q95"] / 1e6,
                    alpha=0.28, color=COLOR_BANDA, label="Q05–Q95 (90%)")
    ax.plot(hs, resultado["q05"] / 1e6,
            color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7, label="Q05 / Q95 (borde)")
    ax.plot(hs, resultado["q95"] / 1e6,
            color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
    ax.plot(hs, resultado["q50"] / 1e6,
            color="crimson", lw=2.0, label="Mediana predicha (Q50)", zorder=5)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlim(hs.min() - 1, hs.max() + 1)


def graficar(resultado: pd.DataFrame, fecha_origen: pd.Timestamp, banco: str,
             idx: int, total: int):
    hs        = resultado["h"].values
    realizado = resultado["target"].values / 1e6
    mask_real = ~np.isnan(realizado)
    h_max_real = int(resultado.loc[mask_real, "h"].max()) if mask_real.any() else 0

    tipo_label = "con Overlay Sobreencaje" if MOSTRAR_OVERLAY else "Base (sin overlay)"
    titulo_base = (
        f"Fan Chart XGBoost QT — {banco}  [{tipo_label}]  |  "
        f"Fecha de origen: {fecha_origen.strftime('%d %b %Y')}  |  "
        f"h = 1 … {int(hs.max())} días hábiles\n"
        f"Realizado disponible: h = 1 … {h_max_real}  |  "
        f"Proyección pura: h = {h_max_real + 1} … {int(hs.max())}"
    )

    cum_q01  = np.cumsum(resultado["q01"].values / 1e6)
    cum_q05  = np.cumsum(resultado["q05"].values / 1e6)
    cum_q50  = np.cumsum(resultado["q50"].values / 1e6)
    cum_q95  = np.cumsum(resultado["q95"].values / 1e6)
    cum_q99  = np.cumsum(resultado["q99"].values / 1e6)
    cum_real = np.where(mask_real, np.nancumsum(np.where(mask_real, realizado, 0)), np.nan)
    cum_real[~mask_real] = np.nan

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), sharex=True,
                                        gridspec_kw={"hspace": 0.08})

    # ── Subplot 1: flujo diario — bandas + realizado ───────────────────────
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

    # ── Subplot 2: flujo diario — solo modelo ─────────────────────────────
    _dibujar_bandas(ax2, hs, resultado)
    if h_max_real > 0:
        ax2.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")
    ax2.set_ylabel("Flujo neto D−R (MM USD)", fontsize=11)
    ax2.set_title("Solo proyección del modelo (sin realizado) — mediana visible",
                  fontsize=10, style="italic")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ── Subplot 3: flujo neto acumulado ───────────────────────────────────
    ax3.fill_between(hs, cum_q01, cum_q99,
                     alpha=0.12, color=COLOR_BANDA, label="Q01–Q99 acum. (98%)")
    ax3.fill_between(hs, cum_q05, cum_q95,
                     alpha=0.28, color=COLOR_BANDA, label="Q05–Q95 acum. (90%)")
    ax3.plot(hs, cum_q05, color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
    ax3.plot(hs, cum_q95, color=COLOR_BANDA, lw=1.0, ls=":", alpha=0.7)
    ax3.plot(hs, cum_q50,
             color="crimson", lw=2.0, label="Mediana acumulada (Q50)", zorder=5)
    if mask_real.any():
        ax3.plot(hs[mask_real], cum_real[mask_real],
                 color="black", lw=2, label="Realizado acumulado (D−R)", zorder=6)
        ax3.scatter(hs[mask_real], cum_real[mask_real],
                    color="black", s=18, zorder=7)
    if h_max_real > 0:
        ax3.axvline(h_max_real, color="red", lw=1.2, ls="--", alpha=0.7,
                    label=f"Último dato realizado (h={h_max_real})")
    ax3.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
    ax3.set_xlabel("Horizonte h (días hábiles desde t)", fontsize=11)
    ax3.set_ylabel("Flujo neto acumulado (MM USD)", fontsize=11)
    ax3.set_title("Flujo neto acumulado D−R desde la fecha de origen",
                  fontsize=10, style="italic")
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax3.set_xlim(hs.min() - 1, hs.max() + 1)

    tipo_sfx = "overlay" if MOSTRAR_OVERLAY else "base"
    nombre = f"fanchart_xgb_qt_{banco}_{fecha_origen.strftime('%Y%m%d')}_{tipo_sfx}.png"
    plt.savefig(DIR_OUTPUT / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{idx}/{total}] Guardado: {nombre}")


# ── Exportación Excel ─────────────────────────────────────────────────────────

def _build_bday() -> CustomBusinessDay:
    """Calendario hábil PE+USA idéntico al de step001 (AbstractHolidayCalendar + Easter)."""
    _f0, _f1 = "2009-01-01", "2042-12-31"

    class _PeruCal(AbstractHolidayCalendar):
        rules = [
            Holiday("AnioNuevo",   month=1,  day=1),
            Holiday("JuevesSanto", month=1,  day=1, offset=[Easter(), _Day(-3)]),
            GoodFriday,
            Holiday("Trabajo",     month=5,  day=1),
            Holiday("SanPedro",    month=6,  day=29),
            Holiday("FiestasP1",   month=7,  day=28),
            Holiday("FiestasP2",   month=7,  day=29),
            Holiday("SantaRosa",   month=8,  day=30),
            Holiday("Angamos",     month=10, day=8),
            Holiday("TodosSantos", month=11, day=1),
            Holiday("Inmaculada",  month=12, day=8),
            Holiday("Nochebuena",  month=12, day=24),
            Holiday("Navidad",     month=12, day=25),
        ]

    hols = set(_PeruCal().holidays(_f0, _f1).normalize())
    hols |= set(USFederalHolidayCalendar().holidays(_f0, _f1).normalize())
    return CustomBusinessDay(holidays=sorted(hols))


def _calc_fecha_th(df: pd.DataFrame, bday: CustomBusinessDay) -> pd.DataFrame:
    """
    Agrega columna fecha_th a df.
    Para cada fecha_t única genera los h_max días hábiles futuros y los mapea
    a cada fila por su valor de h.
    """
    h_max = int(df["h"].max())
    th_map: dict = {}
    for t in df["fecha_t"].unique():
        ts = pd.Timestamp(t)
        fechas = pd.date_range(start=ts + bday, periods=h_max, freq=bday)
        th_map[ts] = {h + 1: fechas[h] for h in range(len(fechas))}

    df = df.copy()
    df["fecha_th"] = df.apply(
        lambda r: th_map.get(r["fecha_t"], {}).get(r["h"], pd.NaT), axis=1
    )
    return df


def _cargar_ambos_parquets(banco: str) -> dict:
    """Carga base y overlay concatenando todos los archivos disponibles."""
    result = {}
    for tipo in ("base", "overlay"):
        candidatos = sorted(DIR_PREDS_STEP005.glob(f"*/preds_{tipo}_{banco}_*.parquet"))
        if not candidatos:
            result[tipo] = None
            continue
        dfs = []
        for ruta in candidatos:
            try:
                _df = pd.read_parquet(ruta)
                _df["fecha_t"] = pd.to_datetime(_df["fecha_t"])
                dfs.append(_df)
            except Exception as e:
                print(f"  [EXCEL/{tipo}] Error leyendo {ruta.name}: {e}")
        if not dfs:
            result[tipo] = None
            continue
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        key_cols = [c for c in ["fecha_t", "h", "banco", "fold"] if c in df.columns]
        df = (df.drop_duplicates(subset=key_cols, keep="last")
                .sort_values(["fecha_t", "h"])
                .reset_index(drop=True))
        result[tipo] = df
        print(f"  [EXCEL/{tipo}] {len(candidatos)} archivo(s) | "
              f"{len(df):,} filas | {df['fecha_t'].nunique()} fechas_t")
    return result


def _preparar_hoja(df: pd.DataFrame, bday: CustomBusinessDay) -> pd.DataFrame:
    """Reordena columnas y convierte a millones USD.

    Usa fecha_th del parquet cuando está disponible (generado por step005 >= v3
    con calendar extendido de step001).  Solo recalcula con bday cuando falta,
    para mantener compatibilidad con parquets antiguos.
    """
    if "fecha_th" not in df.columns or df["fecha_th"].isna().all():
        df = _calc_fecha_th(df, bday)
    else:
        df = df.copy()
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])

    base_cols = ["fecha_t", "fecha_th", "h"]
    if "fold" in df.columns:
        base_cols.append("fold")

    q_cols     = [c for c in ["q01", "q05", "q50", "q95", "q99"] if c in df.columns]
    extra_cols = [c for c in ["mean"] if c in df.columns]
    val_cols   = q_cols + extra_cols + (["target"] if "target" in df.columns else [])

    df = df[base_cols + val_cols].copy()
    for col in val_cols:
        df[col] = df[col] / 1_000_000   # → millones USD

    return df


def _hoja_resumen(df_base, df_overlay) -> pd.DataFrame:
    """Una fila por (fecha_t, fuente) con estadísticas de cobertura."""
    partes = []
    for df, label in [(df_base, "base"), (df_overlay, "overlay")]:
        if df is None or df.empty:
            continue
        g = df.groupby("fecha_t")
        res = pd.DataFrame({
            "fecha_t"       : g["fecha_t"].first(),
            "fecha_th_min"  : g["fecha_th"].min(),
            "fecha_th_max"  : g["fecha_th"].max(),
            "h_min"         : g["h"].min(),
            "h_max"         : g["h"].max(),
            "n_horizontes"  : g["h"].count(),
            "n_realizados"  : g["target"].apply(lambda x: x.notna().sum())
                              if "target" in df.columns else np.nan,
        }).reset_index(drop=True)
        if "target" in df.columns:
            res["pct_realizados"] = (
                res["n_realizados"] / res["n_horizontes"] * 100
            ).round(1)
        if "q50" in df.columns:
            q50_h1 = (df[df["h"] == df["h"].min()]
                      .groupby("fecha_t")["q50"].first())
            res["q50_h_min_MM"] = res["fecha_t"].map(q50_h1)
        res["fuente"] = label
        partes.append(res)
    if not partes:
        return pd.DataFrame()
    return pd.concat(partes, ignore_index=True).sort_values(["fecha_t", "fuente"])


def exportar_excel(banco: str = BANCO, ruta: Path = RUTA_EXCEL) -> None:
    print("\n── Exportación Excel ────────────────────────────────────────────────")
    print("  Construyendo calendario hábil PE+USA...")
    bday = _build_bday()

    print("  Cargando parquets base y overlay...")
    preds = _cargar_ambos_parquets(banco)

    hojas = {}
    for tipo in ("base", "overlay"):
        if preds[tipo] is None:
            print(f"  [{tipo.upper()}] Sin datos — hoja omitida")
            continue
        _has_th = "fecha_th" in preds[tipo].columns and not preds[tipo]["fecha_th"].isna().all()
        print(f"  Preparando hoja '{tipo}' "
              f"({'fecha_th del parquet' if _has_th else 'recalculando fecha_th'})...")
        hojas[tipo] = _preparar_hoja(preds[tipo], bday)

    if not hojas:
        print("  ⚠  Sin datos de predicción. Verificar DIR_PREDS_STEP005.")
        return

    hojas["resumen"] = _hoja_resumen(
        hojas.get("base"),
        hojas.get("overlay"),
    )

    ruta.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Escribiendo Excel → {ruta}")
    with pd.ExcelWriter(ruta, engine="openpyxl",
                        datetime_format="YYYY-MM-DD") as writer:
        for nombre, df_hoja in hojas.items():
            df_hoja.to_excel(writer, sheet_name=nombre, index=False)
            ws = writer.sheets[nombre]
            for col_cells in ws.columns:
                ancho = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(ancho + 2, 28)
            print(f"    Hoja '{nombre}': {df_hoja.shape[0]:,} filas × {df_hoja.shape[1]} cols")

    print(f"  ✓ Excel guardado correctamente.")
    print(f"    Columnas: fecha_t | fecha_th | h | [fold] | Q01..Q99 | target (MM USD)")
    print(f"    Hojas   : base · overlay · resumen")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_preds = cargar_parquet(BANCO)

    todas = fechas_validas(df_preds)
    fechas_selec = todas[::PASO_FECHAS]
    if N_FECHAS_MAX is not None:
        fechas_selec = fechas_selec[:N_FECHAS_MAX]

    tipo_label = "overlay" if MOSTRAR_OVERLAY else "base"
    print(f"\nGenerando {len(fechas_selec)} gráfico(s) [{tipo_label}] "
          f"(paso={PASO_FECHAS} días hábiles, límite={N_FECHAS_MAX}):")

    for i, f_orig in enumerate(fechas_selec, 1):
        fecha_origen = pd.Timestamp(f_orig)
        try:
            resultado = preparar_resultado(df_preds, fecha_origen)
            graficar(resultado, fecha_origen, BANCO, idx=i, total=len(fechas_selec))
        except ValueError as e:
            print(f"  [{i}/{len(fechas_selec)}] Saltando {fecha_origen.date()}: {e}")

    print(f"\n[OK] Todos los gráficos guardados en: {DIR_OUTPUT}")

    if EXPORTAR_EXCEL:
        exportar_excel(banco=BANCO, ruta=RUTA_EXCEL)
