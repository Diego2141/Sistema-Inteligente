# -*- coding: utf-8 -*-
"""
aux_diagnostico_abanico.py
==========================
Diagnóstico: ¿el abanico de predicciones debería abrirse con h?

Compara:
  1. Varianza REAL del target por horizonte   → lo que los datos dicen
  2. Ancho medio del intervalo [Q05-Q95]      → lo que el modelo produce
  3. Pinball loss por horizonte               → calibración efectiva
  4. Fan desde una fecha_t fija              → apertura visual del modelo

Ejecutar:
    python aux_diagnostico_abanico.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_SISTEMA   = Path(r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente")
DIR_PREDS      = BASE_SISTEMA / "2. Output" / "step005_wfcv_v3"
DIR_OUTPUT     = BASE_SISTEMA / "2. Output" / "aux_diagnostico_abanico"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

BANCO          = "SISTEMA"   # nombre tal como aparece en el filename del parquet
DIR_MODO       = None        # None → usa el subdirectorio más reciente con preds_base/overlay
                             # Ejemplo: "xgb_qt_expanding_311"
TIPO_PARQUET   = None        # None → prefiere "overlay", luego "base" (no fold-específicos)
FECHA_T_FIJA   = None        # None → usa una fecha_t del cuarto final del rango


# ── Helpers ───────────────────────────────────────────────────────────────────
def pinball(y: np.ndarray, yhat: np.ndarray, tau: float) -> float:
    err = y - yhat
    return float(np.where(err >= 0, tau * err, (tau - 1) * err).mean())


def cargar_preds(banco: str, tipo: str | None = None) -> pd.DataFrame:
    """Carga el parquet de predicciones consolidadas (todos los folds juntos).

    Lógica de selección:
      1. Si DIR_MODO está definido, busca solo en ese subdirectorio.
      2. Prefiere preds_overlay_* y preds_base_* (todos los folds concatenados).
         Descarta preds_test_fold* (son por fold individual, sin cubrir todos los h).
      3. De los candidatos válidos, usa el más reciente (mayor fecha en el nombre).
      4. Si tipo está especificado, filtra además por "overlay" o "base".
    """
    raiz = DIR_PREDS / DIR_MODO if DIR_MODO else DIR_PREDS
    todos = sorted(raiz.rglob("*.parquet"))

    # Preferir archivos consolidados (preds_base / preds_overlay), descartar fold-específicos
    consolidados = [
        p for p in todos
        if banco.lower() in p.name.lower()
        and "fold" not in p.name.lower()
        and ("preds_base" in p.name or "preds_overlay" in p.name)
    ]

    if tipo:
        consolidados = [p for p in consolidados if tipo in p.name]

    if not consolidados:
        # Fallback: cualquier parquet con el nombre del banco
        consolidados = [p for p in todos if banco.lower() in p.name.lower()]
        if tipo:
            consolidados = [p for p in consolidados if tipo in p.name]

    if not consolidados:
        raise FileNotFoundError(
            f"No se encontró ningún parquet con '{banco}' en {raiz}.\n"
            f"Ajusta BANCO (actualmente '{banco}') o DIR_MODO."
        )

    ruta = consolidados[-1]  # más reciente por orden alfabético (fecha en nombre)
    print(f"[INFO] Usando: {ruta.relative_to(DIR_PREDS)}")
    df = pd.read_parquet(ruta)
    df["fecha_t"]  = pd.to_datetime(df["fecha_t"])
    if "fecha_th" in df.columns:
        df["fecha_th"] = pd.to_datetime(df["fecha_th"])
    return df


# ── Diagnóstico principal ─────────────────────────────────────────────────────
def main() -> None:
    preds = cargar_preds(BANCO, TIPO_PARQUET)  # TIPO_PARQUET=None → autodetecta
    preds = preds.dropna(subset=["target", "q05", "q95", "q50"])

    print(f"[INFO] Columnas disponibles: {list(preds.columns)}")
    print(f"[INFO] Filas cargadas: {len(preds):,}")
    print(f"[INFO] Horizontes disponibles: h={preds['h'].min()}–{preds['h'].max()}")
    print(f"[INFO] Fechas_t: {preds['fecha_t'].min().date()} → {preds['fecha_t'].max().date()}")

    # Ancho del intervalo y pinball por horizonte
    preds["ancho_90"] = preds["q95"] - preds["q05"]
    tiene_q01q99 = "q01" in preds.columns and "q99" in preds.columns
    if tiene_q01q99:
        preds["ancho_98"] = preds["q99"] - preds["q01"]

    agg_dict = dict(
        var_target   = ("target", "var"),
        std_target   = ("target", "std"),
        ancho_90_med = ("ancho_90", "mean"),
        n            = ("target", "count"),
    )
    if tiene_q01q99:
        agg_dict["ancho_98_med"] = ("ancho_98", "mean")

    agg = preds.groupby("h").agg(**agg_dict)

    # Pinball loss por horizonte (q05, q50, q95)
    pb_q05 = preds.groupby("h").apply(
        lambda g: pinball(g["target"].values, g["q05"].values, 0.05)
    ).rename("pb_q05")
    pb_q50 = preds.groupby("h").apply(
        lambda g: pinball(g["target"].values, g["q50"].values, 0.50)
    ).rename("pb_q50")
    pb_q95 = preds.groupby("h").apply(
        lambda g: pinball(g["target"].values, g["q95"].values, 0.95)
    ).rename("pb_q95")

    agg = agg.join(pd.concat([pb_q05, pb_q50, pb_q95], axis=1))

    # Cobertura empírica 90%
    preds["dentro_90"] = (preds["target"] >= preds["q05"]) & (preds["target"] <= preds["q95"])
    cob_90 = preds.groupby("h")["dentro_90"].mean().rename("cobertura_90")
    agg = agg.join(cob_90)

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Diagnóstico de apertura del abanico — {BANCO}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    hs = agg.index.values

    # Panel 1: Varianza real del target vs ancho del modelo
    ax1 = fig.add_subplot(gs[0, :])
    ax1_r = ax1.twinx()
    ax1.plot(hs, np.sqrt(agg["var_target"]),  color="#2563EB", lw=2,   label="σ real target")
    ax1.fill_between(hs,
                     np.sqrt(agg["var_target"]) * 0.8,
                     np.sqrt(agg["var_target"]) * 1.2,
                     alpha=0.15, color="#2563EB")
    ax1_r.plot(hs, agg["ancho_90_med"] / 2, color="#DC2626", lw=2, ls="--", label="½ ancho Q05-Q95 (modelo)")
    if "ancho_98_med" in agg.columns:
        ax1_r.plot(hs, agg["ancho_98_med"] / 2, color="#F97316", lw=1, ls=":", label="½ ancho Q01-Q99 (modelo)")
    ax1.set_xlabel("Horizonte h (días hábiles)")
    ax1.set_ylabel("σ real target (MM USD)", color="#2563EB")
    ax1_r.set_ylabel("½ ancho modelo (MM USD)", color="#DC2626")
    ax1.set_title("¿El modelo abre el abanico como los datos sugieren?", fontweight="bold")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

    ratio_text = ""
    if agg["std_target"].iloc[-1] > 0 and agg["std_target"].iloc[0] > 0:
        ratio_data  = agg["std_target"].iloc[-1] / agg["std_target"].iloc[0]
        ratio_model = (agg["ancho_90_med"].iloc[-1] / agg["ancho_90_med"].iloc[0]
                       if agg["ancho_90_med"].iloc[0] > 0 else float("nan"))
        ratio_text = (
            f"Ratio σ(h_max)/σ(h_min): datos={ratio_data:.2f}x  |  modelo={ratio_model:.2f}x\n"
            f"{'→ Abanico subrepresentado (plano vs datos)' if ratio_data > ratio_model * 1.2 else '→ Modelo aproximadamente bien calibrado en apertura'}"
        )
    ax1.text(0.01, -0.18, ratio_text, transform=ax1.transAxes,
             fontsize=9, color="#374151", style="italic")

    # Panel 2: Pinball loss por horizonte
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hs, agg["pb_q50"], color="#2563EB", lw=2, label="q50")
    ax2.plot(hs, agg["pb_q05"], color="#9333EA", lw=1.5, ls="--", label="q05")
    ax2.plot(hs, agg["pb_q95"], color="#EA580C", lw=1.5, ls="--", label="q95")
    ax2.set_xlabel("Horizonte h")
    ax2.set_ylabel("Pinball loss (MM USD)")
    ax2.set_title("Pinball loss por horizonte", fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel 3: Cobertura empírica 90%
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axhline(0.90, color="#6B7280", lw=1.5, ls="--", label="Objetivo 90%")
    ax3.plot(hs, agg["cobertura_90"], color="#059669", lw=2)
    ax3.fill_between(hs, agg["cobertura_90"], 0.90,
                     where=agg["cobertura_90"] < 0.90,
                     alpha=0.25, color="#DC2626", label="Bajo objetivo")
    ax3.fill_between(hs, agg["cobertura_90"], 0.90,
                     where=agg["cobertura_90"] >= 0.90,
                     alpha=0.2, color="#059669", label="Sobre objetivo")
    ax3.set_xlabel("Horizonte h")
    ax3.set_ylabel("Cobertura empírica")
    ax3.set_ylim(0.5, 1.05)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.set_title("Cobertura empírica [Q05-Q95] por h", fontweight="bold")
    ax3.legend(fontsize=9)

    # Panel 4: Fan desde fecha_t fija
    ax4 = fig.add_subplot(gs[2, :])
    fecha_t_disponibles = preds["fecha_t"].sort_values().unique()
    fecha_ref = (
        pd.Timestamp(FECHA_T_FIJA) if FECHA_T_FIJA
        else fecha_t_disponibles[-max(2, len(fecha_t_disponibles) // 5)]
    )
    # Busca la fecha_t más cercana disponible
    idx_ref = np.searchsorted(fecha_t_disponibles, fecha_ref)
    idx_ref = min(idx_ref, len(fecha_t_disponibles) - 1)
    fecha_ref = pd.Timestamp(fecha_t_disponibles[idx_ref])

    snap = preds[preds["fecha_t"] == fecha_ref].sort_values("h")
    if snap.empty:
        ax4.text(0.5, 0.5, "Sin datos para la fecha de referencia",
                 ha="center", va="center", transform=ax4.transAxes)
    else:
        if tiene_q01q99:
            ax4.fill_between(snap["h"], snap["q01"], snap["q99"],
                             alpha=0.12, color="#2563EB", label="Q01-Q99")
        ax4.fill_between(snap["h"], snap["q05"], snap["q95"],
                         alpha=0.30, color="#2563EB", label="Q05-Q95")
        ax4.plot(snap["h"], snap["q50"],  color="#1D4ED8", lw=2,   label="Q50 (mediana)")
        if "mean" in snap.columns:
            ax4.plot(snap["h"], snap["mean"], color="#1D4ED8", lw=1.5, ls="--", label="Media")
        if snap["target"].notna().any():
            ax4.scatter(snap["h"], snap["target"],
                        color="#DC2626", s=30, zorder=5, label="Target real")
        ax4.axhline(0, color="#9CA3AF", lw=0.8, ls=":")
        ax4.set_xlabel("Horizonte h (días hábiles)")
        ax4.set_ylabel("Flujo D-R (MM USD)")
        ax4.set_title(
            f"Fan desde fecha_t fija: {fecha_ref.date()}  —  "
            f"¿Se abre el abanico con h?",
            fontweight="bold",
        )
        ax4.legend(fontsize=9, ncol=3)

    ruta_fig = DIR_OUTPUT / f"diagnostico_abanico_{BANCO}.png"
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Figura guardada: {ruta_fig}")

    # ── Resumen en consola ─────────────────────────────────────────────────────
    print("\n── Resumen por horizonte (cada 10 h) ────────────────────────────")
    print(f"{'h':>4} {'σ_real':>10} {'½_ancho_90':>12} {'cob_90':>8} {'pb_q50':>10}")
    print("-" * 50)
    for h in hs[::10]:
        r = agg.loc[h]
        print(
            f"{h:4d} "
            f"{r['std_target']:10.0f} "
            f"{r['ancho_90_med']/2:12.0f} "
            f"{r['cobertura_90']:8.1%} "
            f"{r['pb_q50']:10.0f}"
        )

    # ── CSV de respaldo ────────────────────────────────────────────────────────
    ruta_csv = DIR_OUTPUT / f"diagnostico_abanico_{BANCO}.csv"
    agg.to_csv(ruta_csv, float_format="%.2f")
    print(f"[OK] CSV guardado: {ruta_csv}")


if __name__ == "__main__":
    main()
