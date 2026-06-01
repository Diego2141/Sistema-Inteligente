# -*- coding: utf-8 -*-
"""
aux_depuracion_features.py
Genera un Excel para que el equipo analice y depure la lista de 68 features.
Incluye categoría, descripción, riesgo de multicolinealidad y columnas de decisión.
"""

import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

RUTA_OUTPUT = (r"H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente"
               r"\2. Output\depuracion_features.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# Definición completa de features
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # ── Identidad ─────────────────────────────────────────────────────────────
    ("fecha_t",            "Identidad",           "Sistema",
     "Fecha de origen de la predicción",
     "No aplica", ""),
    ("banco",              "Identidad",           "Sistema",
     "Identificador del banco",
     "No aplica", ""),
    ("h",                  "Horizonte",           "Sistema",
     "Horizonte de predicción en días hábiles (2 a 90)",
     "Baja", ""),
    ("log_h",              "Horizonte",           "Sistema",
     "Logaritmo natural del horizonte h — captura relación no lineal con el horizonte",
     "Alta — log_h = log(h), relación determinista con h",
     "Evaluar si h y log_h son redundantes; XGBoost no requiere transformación log"),

    # ── Bancarios: niveles ────────────────────────────────────────────────────
    ("R_t0",   "Bancarios - Nivel", "Datos bancarios",
     "Retiro realizado en t (conocido hoy con 2d de anticipación)",
     "Media — correlacionado con R_t-1, ma_R_5d",
     "Feature clave; retiro del día actual"),
    ("D_t0",   "Bancarios - Nivel", "Datos bancarios",
     "Depósito realizado en t (conocido hoy con 1d de anticipación)",
     "Media — correlacionado con D_t-1, ma_D_5d",
     "Feature clave; depósito del día actual"),
    ("R_t-1",  "Bancarios - Rezagos", "Datos bancarios",
     "Retiro del banco en t-1",
     "Alta — correlacionado con R_t0, R_t-2, ma_R_5d",
     ""),
    ("D_t-1",  "Bancarios - Rezagos", "Datos bancarios",
     "Depósito del banco en t-1",
     "Alta — correlacionado con D_t0, D_t-2, ma_D_5d",
     ""),
    ("R_t-2",  "Bancarios - Rezagos", "Datos bancarios",
     "Retiro del banco en t-2",
     "Alta — correlacionado con R_t-1, R_t-3",
     ""),
    ("D_t-2",  "Bancarios - Rezagos", "Datos bancarios",
     "Depósito del banco en t-2",
     "Alta — correlacionado con D_t-1, D_t-3",
     ""),
    ("R_t-3",  "Bancarios - Rezagos", "Datos bancarios",
     "Retiro del banco en t-3",
     "Alta — correlacionado con R_t-2, R_t-5",
     ""),
    ("D_t-3",  "Bancarios - Rezagos", "Datos bancarios",
     "Depósito del banco en t-3",
     "Alta — correlacionado con D_t-2, D_t-5",
     ""),
    ("R_t-5",  "Bancarios - Rezagos", "Datos bancarios",
     "Retiro del banco en t-5 (mismo día de la semana anterior)",
     "Media — captura patrón semanal",
     "Útil si hay estacionalidad semanal"),
    ("D_t-5",  "Bancarios - Rezagos", "Datos bancarios",
     "Depósito del banco en t-5 (mismo día de la semana anterior)",
     "Media — captura patrón semanal",
     ""),
    ("R_t-22", "Bancarios - Rezagos", "Datos bancarios",
     "Retiro del banco en t-22 (mismo día del mes anterior)",
     "Baja — captura patrón mensual",
     "Útil si hay estacionalidad mensual fuerte"),
    ("D_t-22", "Bancarios - Rezagos", "Datos bancarios",
     "Depósito del banco en t-22 (mismo día del mes anterior)",
     "Baja — captura patrón mensual",
     ""),

    # ── Bancarios: estadísticos ───────────────────────────────────────────────
    ("sigma_R_5d",  "Bancarios - Estadísticos", "Datos bancarios",
     "Desv. estándar rolling 5d de retiros",
     "Alta — correlacionado con sigma_R_22d, garch_vol",
     ""),
    ("sigma_D_5d",  "Bancarios - Estadísticos", "Datos bancarios",
     "Desv. estándar rolling 5d de depósitos",
     "Alta — correlacionado con sigma_D_22d",
     ""),
    ("ma_R_5d",     "Bancarios - Estadísticos", "Datos bancarios",
     "Media móvil 5d de retiros",
     "Alta — combinación lineal de R_t0 a R_t-4",
     "Redundante si ya se tienen los rezagos"),
    ("ma_D_5d",     "Bancarios - Estadísticos", "Datos bancarios",
     "Media móvil 5d de depósitos",
     "Alta — combinación lineal de D_t0 a D_t-4",
     "Redundante si ya se tienen los rezagos"),
    ("sigma_R_22d", "Bancarios - Estadísticos", "Datos bancarios",
     "Desv. estándar rolling 22d de retiros",
     "Alta — correlacionado con sigma_R_5d",
     ""),
    ("sigma_D_22d", "Bancarios - Estadísticos", "Datos bancarios",
     "Desv. estándar rolling 22d de depósitos",
     "Alta — correlacionado con sigma_D_5d",
     ""),
    ("ma_R_22d",    "Bancarios - Estadísticos", "Datos bancarios",
     "Media móvil 22d de retiros",
     "Alta — correlacionado con ma_R_5d y rezagos",
     ""),
    ("ma_D_22d",    "Bancarios - Estadísticos", "Datos bancarios",
     "Media móvil 22d de depósitos",
     "Alta — correlacionado con ma_D_5d y rezagos",
     ""),
    ("delta_R",     "Bancarios - Estadísticos", "Datos bancarios",
     "Variación diaria de retiros: R_t0 - R(t-1)",
     "Alta — combinación lineal de R_t0 y R_t-1",
     "Redundante si ya están R_t0 y R_t-1"),
    ("delta_D",     "Bancarios - Estadísticos", "Datos bancarios",
     "Variación diaria de depósitos: D_t0 - D(t-1)",
     "Alta — combinación lineal de D_t0 y D_t-1",
     "Redundante si ya están D_t0 y D_t-1"),

    # ── Flujo neto ────────────────────────────────────────────────────────────
    ("sigma_flujo_5d",  "Flujo neto", "Datos bancarios",
     "Desv. estándar rolling 5d del flujo neto D−R",
     "Alta — correlacionado con sigma_flujo_20d y garch_vol",
     "Captura régimen de volatilidad del flujo"),
    ("ma_flujo_5d",     "Flujo neto", "Datos bancarios",
     "Media móvil 5d del flujo neto D−R",
     "Alta — correlacionado con ma_flujo_20d y rezagos",
     ""),
    ("sigma_flujo_20d", "Flujo neto", "Datos bancarios",
     "Desv. estándar rolling 20d del flujo neto D−R",
     "Alta — correlacionado con sigma_flujo_5d y garch_vol",
     ""),
    ("ma_flujo_20d",    "Flujo neto", "Datos bancarios",
     "Media móvil 20d del flujo neto D−R",
     "Alta — correlacionado con ma_flujo_5d",
     ""),
    ("garch_vol",       "Flujo neto", "Datos bancarios",
     "Volatilidad condicional GARCH(1,1) del flujo neto D−R",
     "Alta — correlacionado con sigma_flujo_5d y sigma_flujo_20d",
     "Captura clusters de volatilidad; más sofisticado que rolling std"),

    # ── Confirmados ───────────────────────────────────────────────────────────
    ("R_conf_t1", "Confirmados", "Confirmados operativos",
     "Retiro confirmado para t+1 (aviso real o proxy histórico)",
     "Media — correlacionado con R_t0",
     "Feature de alto valor si se tienen los avisos reales"),
    ("R_conf_t2", "Confirmados", "Confirmados operativos",
     "Retiro confirmado para t+2 (aviso real o proxy histórico)",
     "Media — correlacionado con R_conf_t1",
     ""),
    ("D_conf_t1", "Confirmados", "Confirmados operativos",
     "Depósito confirmado para t+1 (aviso real o proxy histórico)",
     "Media — correlacionado con D_t0",
     ""),

    # ── Mercado financiero ────────────────────────────────────────────────────
    ("VIX",       "Mercado financiero", "Yahoo Finance",
     "Índice de volatilidad implícita S&P 500 — mide estrés global",
     "Alta — correlacionado con VIX_ma22 y delta_VIX",
     ""),
    ("delta_VIX", "Mercado financiero", "Yahoo Finance",
     "Variación diaria del VIX",
     "Alta — derivada de VIX",
     "Redundante si ya está VIX; útil para capturar shocks puntuales"),
    ("VIX_ma22",  "Mercado financiero", "Yahoo Finance",
     "Media móvil 22d del VIX — nivel de estrés de fondo",
     "Alta — combinación lineal de VIX de los últimos 22 días",
     ""),
    ("T10Y",      "Mercado financiero", "Yahoo Finance",
     "Rendimiento bono del Tesoro EE.UU. a 10 años",
     "Media — correlacionado con FED_FUNDS y diferencial_tasas",
     "Proxy de expectativas de inflación y apetito por riesgo global"),

    # ── Tipo de cambio ────────────────────────────────────────────────────────
    ("TC_PEN_USD",   "Tipo de cambio", "BCRP Add-In",
     "Tipo de cambio PEN/USD (promedio compra/venta BCRP)",
     "Media — correlacionado con delta_TC (su primera diferencia)",
     ""),
    ("delta_TC",     "Tipo de cambio", "BCRP Add-In",
     "Variación diaria del tipo de cambio",
     "Alta — derivada de TC_PEN_USD",
     ""),
    ("tc_vol_5d",    "Tipo de cambio", "BCRP Add-In",
     "Volatilidad rolling 5d de retornos del TC",
     "Alta — correlacionado con tc_vol_22d y garch_vol_tc",
     ""),
    ("tc_vol_22d",   "Tipo de cambio", "BCRP Add-In",
     "Volatilidad rolling 22d de retornos del TC",
     "Alta — correlacionado con tc_vol_5d y garch_vol_tc",
     ""),
    ("garch_vol_tc", "Tipo de cambio", "BCRP Add-In",
     "Volatilidad condicional GARCH(1,1) de retornos del TC — detecta estrés cambiario",
     "Alta — correlacionado con tc_vol_5d y tc_vol_22d",
     "Más sofisticado que rolling vol"),

    # ── Riesgo país ───────────────────────────────────────────────────────────
    ("EMBI_PERU",      "Riesgo país", "BCRP Add-In",
     "EMBI Perú — spread soberano en puntos básicos",
     "Media — correlacionado con delta_EMBI y garch_vol_embi",
     ""),
    ("delta_EMBI",     "Riesgo país", "BCRP Add-In",
     "Variación diaria del EMBI Perú",
     "Alta — derivada de EMBI_PERU",
     ""),
    ("garch_vol_embi", "Riesgo país", "BCRP Add-In",
     "Volatilidad condicional GARCH(1,1) del EMBI — detecta estrés político",
     "Alta — correlacionado con EMBI_PERU y delta_EMBI",
     ""),

    # ── Política monetaria ────────────────────────────────────────────────────
    ("TASA_REF_BCRP",    "Política monetaria", "BCRP Add-In",
     "Tasa de referencia del BCRP",
     "Alta — diferencial_tasas = TASA_REF_BCRP − FED_FUNDS",
     ""),
    ("FED_FUNDS",        "Política monetaria", "FRED API",
     "Tasa de política monetaria de la Reserva Federal (EE.UU.)",
     "Alta — diferencial_tasas = TASA_REF_BCRP − FED_FUNDS",
     ""),
    ("diferencial_tasas","Política monetaria", "Calculado",
     "TASA_REF_BCRP − FED_FUNDS — diferencial de tasas PEN/USD",
     "Muy alta — combinación lineal exacta de TASA_REF_BCRP y FED_FUNDS",
     "Incluir solo uno de los tres para evitar multicolinealidad exacta"),

    # ── Calendario: posición en mes ───────────────────────────────────────────
    ("dias_al_cierre_mes",    "Calendario - Mes", "Calendario",
     "Días hábiles restantes hasta fin de mes en t+h",
     "Alta — correlacionado con dias_desde_cierre_mes y pos_en_mes",
     ""),
    ("dias_desde_cierre_mes", "Calendario - Mes", "Calendario",
     "Días hábiles transcurridos desde inicio de mes en t+h",
     "Alta — correlacionado con dias_al_cierre_mes y pos_en_mes",
     "dias_al_cierre + dias_desde_cierre ≈ total_bdays_mes"),
    ("pos_en_mes",            "Calendario - Mes", "Calendario",
     "Posición del día hábil dentro del mes (1=primero)",
     "Alta — correlacionado con dias_al_cierre_mes",
     ""),
    ("total_bdays_mes",       "Calendario - Mes", "Calendario",
     "Total de días hábiles del mes de t+h",
     "Media — varía poco entre meses",
     ""),
    ("is_quincena",           "Calendario - Mes", "Calendario",
     "1 si t+h es día 15 o último hábil del mes",
     "Baja — evento puntual",
     "Captura pagos de quincena"),
    ("is_cierre_encaje",      "Calendario - Mes", "Calendario",
     "1 si t+h está en los últimos 2 días hábiles del mes",
     "Baja — evento puntual",
     "Captura presión de encaje bancario"),

    # ── Calendario: trimestral ────────────────────────────────────────────────
    ("is_penult_bday_trim", "Calendario - Trimestre", "Calendario",
     "1 si t+h es penúltimo día hábil del trimestre",
     "Baja — mutuamente excluyentes entre sí",
     ""),
    ("is_ultimo_bday_trim", "Calendario - Trimestre", "Calendario",
     "1 si t+h es último día hábil del trimestre",
     "Baja — mutuamente excluyentes entre sí",
     ""),
    ("is_1er_bday_trim",    "Calendario - Trimestre", "Calendario",
     "1 si t+h es primer día hábil del trimestre",
     "Baja — mutuamente excluyentes entre sí",
     ""),
    ("is_2do_bday_trim",    "Calendario - Trimestre", "Calendario",
     "1 si t+h es segundo día hábil del trimestre",
     "Baja — mutuamente excluyentes entre sí",
     ""),
    ("is_3er_bday_trim",    "Calendario - Trimestre", "Calendario",
     "1 si t+h es tercer día hábil del trimestre",
     "Baja — mutuamente excluyentes entre sí",
     ""),

    # ── Calendario: general ───────────────────────────────────────────────────
    ("dia_semana",      "Calendario - General", "Calendario",
     "Día de la semana (0=lunes, 4=viernes)",
     "Baja",
     "Captura patrones de inicio/fin de semana"),
    ("mes",             "Calendario - General", "Calendario",
     "Mes del año (1–12)",
     "Baja",
     "Captura estacionalidad anual"),
    ("is_fin_anio",     "Calendario - General", "Calendario",
     "1 si t+h es 28–31 de diciembre",
     "Baja — evento puntual anual",
     ""),
    ("is_pre_feriado",  "Calendario - General", "Calendario / holidays",
     "1 si el día siguiente a t+h es feriado PE o US",
     "Baja",
     ""),
    ("is_post_feriado", "Calendario - General", "Calendario / holidays",
     "1 si el día anterior a t+h es feriado PE o US",
     "Baja",
     ""),
    ("is_pre_eleccion", "Calendario - General", "Calendario",
     "1 si t+h está dentro de los 7 días previos a elecciones presidenciales",
     "Baja — evento muy esporádico",
     ""),
    ("is_post_eleccion","Calendario - General", "Calendario",
     "1 si t+h está dentro de los 7 días posteriores a elecciones presidenciales",
     "Baja — evento muy esporádico",
     ""),

    # ── Target ────────────────────────────────────────────────────────────────
    ("target", "Target", "Datos bancarios",
     "Flujo neto = D(b, t+h) - R(b, t+h). Variable a predecir.",
     "No aplica", ""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Colores por categoría
# ─────────────────────────────────────────────────────────────────────────────
COLORES_CAT = {
    "Identidad":                  "D9D9D9",
    "Horizonte":                  "F2F2F2",
    "Bancarios - Nivel":          "DEEAF1",
    "Bancarios - Rezagos":        "BDD7EE",
    "Bancarios - Estadísticos":   "9DC3E6",
    "Flujo neto":                 "2E75B6",
    "Confirmados":                "FCE4D6",
    "Mercado financiero":         "E2EFDA",
    "Tipo de cambio":             "C6EFCE",
    "Riesgo país":                "FFEB9C",
    "Política monetaria":         "FFC7CE",
    "Calendario - Mes":           "EAD1DC",
    "Calendario - Trimestre":     "D5B8E0",
    "Calendario - General":       "C9B8E8",
    "Target":                     "404040",
}

COLORES_RIESGO = {
    "No aplica": "FFFFFF",
    "Baja":      "C6EFCE",
    "Media":     "FFEB9C",
    "Alta":      "FFCC99",
    "Muy alta":  "FFC7CE",
}


def build_df():
    rows = []
    for (var, cat, fuente, desc, riesgo, notas) in FEATURES:
        nivel_riesgo = riesgo.split(" —")[0].strip()
        rows.append({
            "Variable":                var,
            "Categoría":               cat,
            "Fuente":                  fuente,
            "Descripción":             desc,
            "Riesgo multicolinealidad": nivel_riesgo,
            "Detalle multicolinealidad": riesgo,
            "Notas técnicas":          notas,
            "Decisión":                "",
            "Justificación equipo":    "",
        })
    return pd.DataFrame(rows)


def apply_format(path):
    wb = load_workbook(path)
    ws = wb.active
    ws.title = "Depuración Features"

    # ── Anchos de columna ─────────────────────────────────────────────────────
    anchos = {
        "A": 22,   # Variable
        "B": 26,   # Categoría
        "C": 24,   # Fuente
        "D": 60,   # Descripción
        "E": 14,   # Riesgo
        "F": 55,   # Detalle multicolinealidad
        "G": 55,   # Notas técnicas
        "H": 14,   # Decisión
        "I": 45,   # Justificación
    }
    for col, ancho in anchos.items():
        ws.column_dimensions[col].width = ancho

    # ── Encabezado ────────────────────────────────────────────────────────────
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_border = Border(
        bottom=Side(style="medium", color="FFFFFF")
    )
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border = header_border
    ws.row_dimensions[1].height = 30

    # ── Cuerpo ────────────────────────────────────────────────────────────────
    thin = Side(style="thin", color="CCCCCC")
    border_thin = Border(left=thin, right=thin, top=thin, bottom=thin)

    df = build_df()
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
        cat = df.iloc[row_idx - 2]["Categoría"]
        riesgo = df.iloc[row_idx - 2]["Riesgo multicolinealidad"]

        cat_color   = COLORES_CAT.get(cat, "FFFFFF")
        riesgo_color = COLORES_RIESGO.get(riesgo, "FFFFFF")

        # Color de fuente claro para "Target" y "Flujo neto"
        font_color = "FFFFFF" if cat in ("Target", "Flujo neto") else "000000"

        for cell in row:
            col_letter = get_column_letter(cell.column)
            cell.border = border_thin
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.font = Font(color=font_color, size=10)

            if col_letter == "E":
                cell.fill = PatternFill("solid", fgColor=riesgo_color)
                cell.font = Font(color="000000", size=10, bold=True)
                cell.alignment = Alignment(horizontal="center",
                                           vertical="center")
            elif col_letter == "H":
                # Columna Decisión: fondo blanco, negrita
                cell.fill = PatternFill("solid", fgColor="FFFACD")
                cell.font = Font(color="000000", size=10, bold=True)
                cell.alignment = Alignment(horizontal="center",
                                           vertical="center")
            elif col_letter in ("A", "B"):
                cell.fill = PatternFill("solid", fgColor=cat_color)
            else:
                cell.fill = PatternFill("solid", fgColor=cat_color
                                        if col_letter not in ("F", "G", "I")
                                        else "FAFAFA")

        ws.row_dimensions[row_idx].height = 40

    # ── Validación dropdown en columna Decisión ───────────────────────────────
    dv = DataValidation(
        type="list",
        formula1='"Mantener,Eliminar,Evaluar,Agregar variante"',
        allow_blank=True,
        showDropDown=False,
    )
    dv.sqref = f"H2:H{ws.max_row}"
    ws.add_data_validation(dv)

    # ── Fijar encabezado ──────────────────────────────────────────────────────
    ws.freeze_panes = "A2"

    # ── Leyenda en hoja separada ──────────────────────────────────────────────
    ws_ley = wb.create_sheet("Leyenda")
    ws_ley.column_dimensions["A"].width = 30
    ws_ley.column_dimensions["B"].width = 55

    leyenda = [
        ("LEYENDA — RIESGO MULTICOLINEALIDAD", ""),
        ("Baja",      "Feature independiente; bajo riesgo de redundancia"),
        ("Media",     "Cierta correlación con otros features; vigilar importancia"),
        ("Alta",      "Correlacionado con varios features del mismo grupo; evaluar si aporta información nueva"),
        ("Muy alta",  "Relación matemática casi exacta con otros features; considerar eliminar"),
        ("", ""),
        ("OPCIONES DE DECISIÓN", ""),
        ("Mantener",         "Feature validado; incluir en el modelo"),
        ("Eliminar",         "Feature redundante o sin valor predictivo; excluir"),
        ("Evaluar",          "Requiere análisis adicional (importancia, correlación, dominio)"),
        ("Agregar variante", "Reemplazar por una versión mejorada o un feature relacionado"),
        ("", ""),
        ("GUÍA DE ANÁLISIS", ""),
        ("Paso 1", "Identificar grupos de alta multicolinealidad y elegir el representante más informativo"),
        ("Paso 2", "Revisar importancia (gain) por fold en step005 — features con importancia muy baja o inestable son candidatos a eliminar"),
        ("Paso 3", "Consultar con expertos de dominio si el feature tiene lógica económica/operativa"),
        ("Paso 4", "Re-correr step005 con la lista reducida y comparar cobertura en heatmap"),
    ]

    header_ley = PatternFill("solid", fgColor="1F4E79")
    for i, (col_a, col_b) in enumerate(leyenda, start=1):
        ws_ley.cell(i, 1, col_a)
        ws_ley.cell(i, 2, col_b)
        if col_b == "" and col_a:
            for c in [ws_ley.cell(i, 1), ws_ley.cell(i, 2)]:
                c.fill = header_ley
                c.font = Font(bold=True, color="FFFFFF", size=11)
        ws_ley.row_dimensions[i].height = 22

    for nivel, color in COLORES_RIESGO.items():
        if nivel == "No aplica":
            continue
        for row in ws_ley.iter_rows(min_row=1, max_row=ws_ley.max_row):
            if row[0].value == nivel:
                row[0].fill = PatternFill("solid", fgColor=color)
                row[0].font = Font(bold=True, size=10)

    wb.save(path)
    print(f"Archivo guardado en: {path}")


if __name__ == "__main__":
    Path(RUTA_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df = build_df()
    df.to_excel(RUTA_OUTPUT, index=False)
    apply_format(RUTA_OUTPUT)
    print(f"\nTotal features: {len(df)}")
    print("\nResumen por categoría:")
    print(df["Categoría"].value_counts().to_string())
    print("\nResumen por riesgo de multicolinealidad:")
    print(df["Riesgo multicolinealidad"].value_counts().to_string())
