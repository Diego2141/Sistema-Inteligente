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

# Matriz de features (step001) — fuente autoritativa de fecha_th con calendario extendido.
# Confidencial: no se commitea (.gitignore). Se lee solo para extraer fecha_th.
RUTA_MATRIZ = BASE_SISTEMA / "1. Data" / "Clean" / "matriz_features.parquet"

BANCO = "SISTEMA"

# ── Exportación Excel ─────────────────────────────────────────────────────────
EXPORTAR_EXCEL = True   # True = genera Excel al final del script; False = solo PNGs
RUTA_EXCEL = (BASE_SISTEMA / "2. Output" / "aux_fanchart_horizontes_xgb_qt"
              / "fanchart_datos.xlsx")

# ── Parámetros del loop ───────────────────────────────────────────────────────
PASO_FECHAS    = 2     # cada 2 días hábiles; usar 1 para todas las fechas
N_FECHAS_MAX   = None  # None = generar todos; ej: 6 para las 6 primeras fechas válidas
MOSTRAR_OVERLAY = True  # False = predicciones base; True = con overlay sobreencaje

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


def _cargar_mapping_from_matriz(banco: str) -> "pd.DataFrame | None":
    """
    Construye el mapping (fecha_t × h → fecha_th, target) desde la matriz de
    features de step001.  Retorna DataFrame [fecha_t, h, fecha_th, target] o None.

    Por qué reemplazar AMBAS columnas (fecha_th y target) desde la matriz:
    ──────────────────────────────────────────────────────────────────────
    El parquet de predicciones (step005) puede haber sido generado con una versión
    anterior de la matriz de features (con calendario distinto).  En ese caso:
      · target en parquet = D(fecha_th_antigua) − R(fecha_th_antigua)
      · fecha_th_nueva   ≠ fecha_th_antigua  →  target incorrecto para la nueva fecha
    Mostrar la fecha_th correcta con el target viejo produce múltiples valores de
    target para el mismo fecha_th al filtrar en Excel.

    La solución: tomar SIEMPRE fecha_th y target de la MISMA fuente (la matriz
    actual), garantizando coherencia total: target = D(fecha_th) − R(fecha_th).

    Estrategia para fecha_th (en orden de prioridad):
      1. Columna fecha_th en la matriz (disponible si step001 >= commit 2165-2168).
      2. Derivación desde la secuencia de fecha_t únicas: esas fechas son
         exactamente los días hábiles del calendario extendido de step001
         (PE+USA + ~44 días no hábiles auto-detectados).
         fecha_th(fecha_t, h) = dias_habiles[pos(fecha_t) + h]
         → matemáticamente equivalente al cálculo de step001 sin ningún calendario.
    """
    if not RUTA_MATRIZ.exists():
        print(f"  [AVISO] Matriz de features no encontrada en {RUTA_MATRIZ}. "
              f"Se usará calendario básico (puede haber shift de 1 día en puentes).")
        return None

    # ── Leer target y fecha_t, h siempre ────────────────────────────────────
    cols_base = ["fecha_t", "h", "target"]
    try:
        df_base = pd.read_parquet(
            RUTA_MATRIZ,
            columns=cols_base,
            filters=[("banco", "==", banco)],
        )
    except Exception as e:
        print(f"  [AVISO] No se pudo leer la matriz de features: {e}. "
              f"Se usará calendario básico.")
        return None

    df_base["fecha_t"] = pd.to_datetime(df_base["fecha_t"])
    df_base["h"]       = df_base["h"].astype(int)
    # target puede tener NaT en filas sin datos realizados (futuro); las conservamos
    df_base = df_base.drop_duplicates(subset=["fecha_t", "h"])

    # ── Intento 1: leer fecha_th directamente ────────────────────────────────
    _fth_ok = False
    try:
        df_fth = pd.read_parquet(
            RUTA_MATRIZ,
            columns=["fecha_t", "h", "fecha_th"],
            filters=[("banco", "==", banco)],
        )
        df_fth["fecha_t"] = pd.to_datetime(df_fth["fecha_t"])
        _fth_col = pd.to_datetime(df_fth["fecha_th"])
        if getattr(_fth_col.dt, "tz", None) is not None:
            _fth_col = _fth_col.dt.tz_convert(None)
        df_fth["fecha_th"] = _fth_col
        df_fth["h"]        = df_fth["h"].astype(int)
        df_fth = df_fth.dropna(subset=["fecha_th"])
        if not df_fth.empty:
            df_fth = df_fth[["fecha_t", "h", "fecha_th"]].drop_duplicates(subset=["fecha_t", "h"])
            df_base = df_base.merge(df_fth, on=["fecha_t", "h"], how="left")
            _fth_ok = True
            print(f"  [MATRIZ] fecha_th leída directamente de la columna.")
    except Exception:
        pass   # columna fecha_th no existe → intento 2

    # ── Intento 2: derivar fecha_th desde la secuencia de fecha_t únicas ────
    if not _fth_ok:
        dias_habiles = np.sort(df_base["fecha_t"].unique())
        n_dias = len(dias_habiles)
        if n_dias == 0:
            print("  [AVISO] La matriz de features no contiene filas para este banco.")
            return None

        pos_dict = {pd.Timestamp(d): i for i, d in enumerate(dias_habiles)}

        # Vectorizado: fecha_th = dias_habiles[pos(fecha_t) + h]
        ft_idx = df_base["fecha_t"].map(pos_dict)
        th_idx = ft_idx + df_base["h"]
        valido = ft_idx.notna() & (th_idx < n_dias)

        df_base["fecha_th"] = pd.NaT
        if valido.any():
            df_base.loc[valido, "fecha_th"] = pd.DatetimeIndex(
                dias_habiles[th_idx[valido].astype(int).values]
            )
        df_base["fecha_th"] = pd.to_datetime(df_base["fecha_th"])

        rango = (f"{pd.Timestamp(dias_habiles[0]).date()} → "
                 f"{pd.Timestamp(dias_habiles[-1]).date()}")
        n_ok = int(df_base["fecha_th"].notna().sum())
        print(f"  [MATRIZ] fecha_th derivada de secuencia hábil: "
              f"{n_ok:,}/{len(df_base):,} pares | {n_dias} días hábiles ({rango})")

    result = df_base[["fecha_t", "h", "fecha_th", "target"]].copy()
    result = result.dropna(subset=["fecha_th"])
    print(f"  [MATRIZ] mapping final: {len(result):,} pares (fecha_t, h) con "
          f"fecha_th y target de la matriz actual")
    return result if not result.empty else None


def _calc_fecha_th(df: pd.DataFrame, bday: CustomBusinessDay) -> pd.DataFrame:
    """
    Agrega / recalcula columna fecha_th con dtype datetime64[ns].
    Usa merge vectorizado en lugar de apply para garantizar el dtype correcto
    (apply con Timestamps produce object, que openpyxl no escribe como fecha).
    """
    h_max = int(df["h"].max())
    fechas_t_unicas = pd.DatetimeIndex(df["fecha_t"].unique())

    records = []
    for t in fechas_t_unicas:
        fechas = pd.date_range(start=t + bday, periods=h_max, freq=bday)
        for h_idx, fth in enumerate(fechas, start=1):
            records.append((t, h_idx, fth))

    th_df = pd.DataFrame(records, columns=["fecha_t", "h", "fecha_th"])
    th_df["fecha_th"] = pd.to_datetime(th_df["fecha_th"])        # datetime64[ns]
    th_df["h"] = th_df["h"].astype(int)

    df = df.copy()
    # Drop fecha_th si ya existe (viene como NaT desde el parquet)
    df = df.drop(columns=["fecha_th"], errors="ignore")
    h_col = df["h"].astype(int)
    df["h"] = h_col
    df = df.merge(th_df, on=["fecha_t", "h"], how="left")
    return df


def _cargar_ambos_parquets(banco: str) -> dict:
    """Carga base y overlay del parquet más reciente de step005.

    fecha_th y target se toman SIEMPRE de la matriz de features (step001),
    no del parquet de predicciones.  Esto garantiza que target = D(fecha_th) − R(fecha_th)
    y que todas las filas apuntando al mismo fecha_th tengan el mismo target.

    Los cuantiles (q01, q05, q50, q95, q99, mean) se toman del parquet de predicciones.
    """
    # Cargar mapping (fecha_t, h → fecha_th, target) de la matriz una sola vez
    _mapping = _cargar_mapping_from_matriz(banco)

    result = {}
    for tipo in ("base", "overlay"):
        candidatos = sorted(DIR_PREDS_STEP005.glob(f"*/preds_{tipo}_{banco}_*.parquet"))
        if not candidatos:
            result[tipo] = None
            continue
        # Usar solo el parquet más reciente: cada run de step005 contiene el histórico
        # completo; acumular runs mezclaría fecha_th de calendarios distintos.
        ruta = candidatos[-1]
        try:
            df = pd.read_parquet(ruta)
            df["fecha_t"] = pd.to_datetime(df["fecha_t"])
            if "fecha_th" in df.columns:
                _fth = pd.to_datetime(df["fecha_th"])
                if getattr(_fth.dt, "tz", None) is not None:
                    _fth = _fth.dt.tz_convert(None)
                df["fecha_th"] = _fth
            else:
                df["fecha_th"] = pd.NaT
        except Exception as e:
            print(f"  [EXCEL/{tipo}] Error leyendo {ruta.name}: {e}")
            result[tipo] = None
            continue

        # Para cada (fecha_t, h, banco), conservar solo el fold más reciente.
        dedup2 = [c for c in ["fecha_t", "h", "banco"] if c in df.columns]
        if "fold" in df.columns and dedup2:
            df = (df.sort_values(dedup2 + ["fold"])
                    .drop_duplicates(subset=dedup2, keep="last"))
        df = df.sort_values(["fecha_t", "h"]).reset_index(drop=True)

        # ── Reemplazar fecha_th y target desde la matriz (fuente autoritativa) ──
        # El parquet de predicciones puede tener targets stale (calculados con un
        # calendario antiguo donde fecha_th era distinta).  Mostrar la fecha_th
        # correcta con el target incorrecto produce múltiples targets para la misma
        # fecha en Excel.  Solución: tomar AMBAS columnas de la matriz actual.
        #
        # Filas cuya fecha_t no está en la matriz son predicciones de un run de
        # step005 con un calendario que ya no coincide; se eliminan.
        if _mapping is not None:
            _valid_ft = set(_mapping["fecha_t"].unique())
            _stale_mask = ~df["fecha_t"].isin(_valid_ft)
            if _stale_mask.any():
                n_stale = int(_stale_mask.sum())
                ft_stale = sorted(df.loc[_stale_mask, "fecha_t"].dt.date.unique())
                print(f"  [EXCEL/{tipo}] Eliminando {n_stale} filas con fecha_t "
                      f"no hábil según la matriz actual: {ft_stale}")
                df = df[~_stale_mask].copy()

            df["h"] = df["h"].astype(int)
            # Eliminar columnas que vienen de la matriz (se reemplazarán)
            df = df.drop(columns=["fecha_th", "target", "target_t0"], errors="ignore")
            df = df.merge(_mapping, on=["fecha_t", "h"], how="left")

            # ── target_t0: valor realizado D-R en la fecha de ORIGEN (fecha_t) ──
            # Construir lookup {fecha → realized_DR} desde los pares (fecha_th, target)
            # del mapping. Las fechas de origen son días hábiles que también aparecen
            # como fechas pronosticadas en filas de otras fechas anteriores.
            _realized = (
                _mapping.dropna(subset=["target"])
                .drop_duplicates(subset=["fecha_th"])
                .set_index("fecha_th")["target"]
            )
            df["target_t0"] = df["fecha_t"].map(_realized)

            n_fth   = int(df["fecha_th"].notna().sum())
            n_tgt   = int(df["target"].notna().sum())
            n_t0    = int(df["target_t0"].notna().sum())
            n_total = len(df)
            print(f"  [EXCEL/{tipo}] fecha_th: {n_fth:,}/{n_total:,} | "
                  f"target: {n_tgt:,}/{n_total:,} | "
                  f"target_t0: {n_t0:,}/{n_total:,}"
                  + (" ✓" if n_fth == n_total else
                     f" ({n_total - n_fth} sin cubrir — h muy grande o fecha futura)"))

        result[tipo] = df
        n_folds = df["fold"].nunique() if "fold" in df.columns else "?"
        print(f"  [EXCEL/{tipo}] {ruta.name} | "
              f"{len(df):,} filas | {df['fecha_t'].nunique()} fechas_t | "
              f"{n_folds} fold(s) en datos")
    return result


def _preparar_hoja(df: pd.DataFrame, bday: CustomBusinessDay) -> pd.DataFrame:
    """Reordena columnas y convierte a millones USD.

    Prioridad de fecha_th (aplicada ANTES de llegar aquí, en _cargar_ambos_parquets):
      1. Parquet step005 (propagado desde step001, >= commit 7827417)
      2. Matriz de features step001 (calendario extendido correcto)
    Si aún quedan NaT, esta función aplica el fallback de último recurso:
      3. _calc_fecha_th con calendario básico (puede dar shift de 1 día en puentes)
    """
    df = df.copy()

    # Convertir / strip-timezone
    if "fecha_th" in df.columns:
        _fth = pd.to_datetime(df["fecha_th"])
        if getattr(_fth.dt, "tz", None) is not None:
            _fth = _fth.dt.tz_convert(None)
        df["fecha_th"] = _fth
    else:
        df["fecha_th"] = pd.NaT

    # Fallback de último recurso: ni el parquet ni la matriz tenían fecha_th.
    # - Si TODOS son NaT: relleno uniforme con calendario básico (sin mezcla).
    # - Si ALGUNOS son NaT: no rellenar para evitar mezcla de calendarios.
    _nat_mask = df["fecha_th"].isna()
    if _nat_mask.all():
        _df_fill = _calc_fecha_th(df.copy(), bday)
        df["fecha_th"] = _df_fill["fecha_th"].values
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])
        print(f"  [AVISO] fecha_th calculada con calendario básico (sin matriz disponible). "
              f"Re-correr step001 para corregir posibles shifts en periodos de puente.")
    elif _nat_mask.any():
        n_nat = int(_nat_mask.sum())
        print(f"  [AVISO] {n_nat} filas con fecha_th=NaT tras enriquecimiento. "
              f"Re-correr step001 + step005 para corregir.")

    base_cols = ["fecha_t", "fecha_th", "h"]
    if "fold" in df.columns:
        base_cols.append("fold")

    q_cols     = [c for c in ["q01", "q05", "q50", "q95", "q99"] if c in df.columns]
    extra_cols = [c for c in ["mean"] if c in df.columns]
    # target    = D(fecha_th) − R(fecha_th): realizado en la fecha PRONOSTICADA
    # target_t0 = D(fecha_t)  − R(fecha_t):  realizado en la fecha de ORIGEN
    tgt_cols = [c for c in ["target", "target_t0"] if c in df.columns]
    val_cols = q_cols + extra_cols + tgt_cols

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
    print(f"    Columnas: fecha_t | fecha_th | h | [fold] | Q01..Q99 | target | target_t0 (MM USD)"
          f"\n    target    = D(fecha_th) − R(fecha_th): realizado en fecha PRONOSTICADA"
          f"\n    target_t0 = D(fecha_t)  − R(fecha_t):  realizado en fecha de ORIGEN")
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
