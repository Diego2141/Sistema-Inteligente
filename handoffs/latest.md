# Handoff — Build feature matrix for liquidity prediction

**Sesión:** 2026-08-24
**Branch:** `claude/build-feature-matrix-HDMPg`
**HEAD al cerrar:** `0de732e` (+ un commit pendiente con el fix del checklist CCOVN, ver "En progreso")
**Working tree:** limpio salvo `aux_verificar_ccovn_particion.py` (fix aplicado, sin commitear)

> No existía `HANDOFF_TEMPLATE.md` en el repo. Este archivo usa la estructura
> pedida en la instrucción.

---

## 1. Tarea de la sesión

Dos hilos que terminaron entrelazados:

1. **Feature engineering de la familia `*_pos`** (rezagos por posición del mes)
   en `step001_build_feature_matrix.py`: corregir que el número de rezagos que
   entra al extremo caía de 4 a 1 con el horizonte.
2. **Particiones del sistema** en una versión nueva, `step001_build_feature_matrix_v2.py`:
   poder modelar `FOCO`/`RESTO` por separado (BBVA contra el resto, o bancos
   globales contra banca local) en vez de solo el agregado SISTEMA.

Contexto de negocio: BBVA concentra ~94% del neto del sistema en la ventana de
cierre, así que el modelo agregado termina ajustando ese banco más ruido.

---

## 2. Estado

### Hecho y verificado

| Qué | Dónde | Verificación |
|---|---|---|
| Ventana deslizante de rezagos (conteo fijo 4, recencia máxima) | `step001_build_feature_matrix.py` | `aux_verificar_pos_ventana.py` pasa |
| Particiones configurables `bbva` / `globales` / `None` | `step001_build_feature_matrix_v2.py` | `aux_verificar_particion.py` pasa (14 secciones) |
| CCOVN con roles relativos propio/contraparte | `step001_build_feature_matrix_v2.py` | `aux_verificar_ccovn_particion.py` pasa |
| Tabla de alias CCOVN + guarda de categorías | `step001_build_feature_matrix_v2.py` | verificado contra los 86 headers reales |
| Orden de columnas determinista (alfabético) | `step001_build_feature_matrix_v2.py` | sección 12 del checklist |
| Exportador Excel por partición con hoja de conciliación | `aux_exportar_matriz_excel_v2.py` | probado end-to-end con matriz sintética |
| `pyflakes` limpio (0 undefined) en v1 y v2 | ambos | ver comandos abajo |

### En progreso

- **Fix sin commitear:** `aux_verificar_ccovn_particion.py` tenía la aserción
  `check("SISTEMA: sin contraparte", cc is None)`, que quedó desactualizada
  cuando SISTEMA pasó a recibir `foco` como contraparte. Ya está corregida y el
  checklist pasa, pero **falta commitear**. Es lo primero a hacer (ver §6).

### Bloqueado / no terminado

- **La corrida completa de v2 nunca llegó al final.** Falló tres veces seguidas
  por incompatibilidad de esquema entre bancos. Los tres bugs están arreglados,
  pero **la última corrida del usuario fue antes del último fix**, así que
  todavía no hay confirmación de que v2 termine y escriba el Parquet.
- **Recombinación FOCO+RESTO → total:** no implementada. Es el paso conceptual
  siguiente y tiene una trampa importante documentada en §4.
- **Diagnóstico de correlación foco/resto:** propuesto, no implementado. Es el
  número que decide si vale la pena la recombinación.

---

## 3. Archivos tocados

### `step001_build_feature_matrix.py` (v1, +115/-21)
- `LAGS_POSICION_MES` pasa de `[1,2,3,4]` a **lista de candidatos** `[1..8]`, más
  `N_REZAGOS_OBJETIVO = 4`.
- `_build_lag_posicion_mes()` acepta `n_objetivo`. Una línea en el bucle:
  `usable &= (n_lags < n_objetivo)`. Como se recorre del rezago más reciente al
  más viejo, entran siempre los 4 más nuevos disponibles.
- La función ordena `lags_meses` internamente (de ese orden depende cuáles
  elige, no se confía en el llamador).
- Warning nuevo si <99% de las filas llega al objetivo.

### `step001_build_feature_matrix_v2.py` (nuevo, 4573 líneas)
Copia de v1 más:
- `PARTICIONES` (`bbva`, `globales`) + `PARAMS["particion_activa"]`.
- `aplicar_particion()`, `columnas_derivadas()`, `resolver_ccovn_lados()`,
  `armar_sub_ccovn()`, `_mapear_bancos_ccovn()`, `destinos_encaje_bbva()`.
- `ALIAS_CCOVN` (10 globales + `BCP→CREDITO`), `CCOVN_NO_BANCOS` (guarda).
- `BBVA_FEAT_COLS` / `BBVA_INTERMEDIAS` / `BBVA_FEAT_FINALES` a nivel de módulo.
- `POLITICA_ENCAJE_BBVA = "exacto_y_sistema"`.
- `frec_flujo_pos` agregada a `FEATURES_EXCLUIR`.
- Orden final de columnas **alfabético** (no de inserción).
- Diagnóstico de esquema en el escritor: distingue diferencia de conjunto
  (error) de diferencia de orden (realinea con warning).

### Checklists nuevos
- `aux_verificar_pos_ventana.py` — ventana deslizante, 8 secciones.
- `aux_verificar_particion.py` — particiones, 14 secciones.
- `aux_verificar_ccovn_particion.py` — roles CCOVN.

### `aux_exportar_matriz_excel_v2.py` (nuevo)
Detecta grupos en la matriz, una hoja por grupo, más `CONCILIACION`
(SISTEMA vs FOCO+RESTO por fecha y h), `RESUMEN_ANUAL`, `RESUMEN_GRUPOS`.

### `onepager_encaje.html`, `diccionario_features.html`
Fuentes de los dos artifacts publicados, subidas al repo.

---

## 4. Decisiones no obvias

**Ventana deslizante en vez de lista fija lejana.**
Se probó primero una familia paralela `*_pos_fx` con rezagos `[4,5,6,7]`
(commit `41a8c13`) y **se revirtió entera** (`bb042d2`). Dos razones: dos
columnas muy correlacionadas se reparten la importancia en permutación y las
dos salen deprimidas, contaminando justo la medición que motivaba el cambio; y
fijar rezagos lejanos pagaba recencia en horizontes cortos, donde no había
problema. La ventana deslizante da conteo fijo **y** recencia máxima sin
columnas nuevas.

**Propiedad a retener:** donde el comportamiento previo ya llegaba a 4 rezagos,
el resultado es **idéntico**. Las mediciones de importancia ya hechas en ese
tramo siguen valiendo.

**La partición corre ANTES de `agrupar_bancos()`.**
Obligatorio, no preferencia. `agrupar_bancos()` suma los chicos en
`Otros_bancos` y **elimina** sus columnas, y varios globales (Deutsche, ICBC,
Bank of China, BCI) están bajo el umbral del 1%. Partir después los dejaría
contados como banca local sin aviso. `aplicar_particion()` aborta con
`RuntimeError` si detecta que se la llamó tarde.

**`resto = sistema − foco`, no suma de emparejados.**
Solo los globales pueden componer un FOCO y todos tienen columna conocida en
`Saldos_CCOVN.xlsx`. `sistema` ya es la suma de todas las columnas sin depender
del matching, así que la diferencia es exacta. Elimina la dependencia del
emparejamiento para los ~20 bancos del resto.
*Contrapartida:* `sistema` cubre las 86 entidades del CCOVN (cajas,
cooperativas, financieras, fondos), un universo más amplio que los 23 bancos de
Transacciones. Entonces `resto` significa "todo lo que no es el foco". Es la
misma definición amplia que `ccovn_sistema_lag1` ya tenía en v1.

**Encaje por composición EXACTA del grupo, no "contiene".**
`FOCO_GLOBALES` contiene a BBVA pero son cinco bancos: atribuirle el encaje de
uno sería inventar un dato para los otros cuatro. La primera versión usaba
"contiene" y le daba las features a `FOCO_GLOBALES`.

**SISTEMA recibe `foco` como contraparte CCOVN.**
Sin eso, `share_propio` da constante 1 y SISTEMA pierde la señal de
concentración que en v1 viajaba como `bbva_share_lag1`, que es la variable del
hallazgo 3. Por eso también se anulan a NaN `share_propio_lag1` y
`var_ccovn_propio_exceso_lag1` cuando `clave_p == "sistema"`: son constantes por
construcción y ensucian los heatmaps de importancia.

**Orden de columnas alfabético.**
No cosmético. En orden de inserción, la posición depende del camino de
ejecución, y el `ParquetWriter` aborta. Alfabético hace el esquema función
únicamente del conjunto de columnas.

---

## 5. Gotchas (cosas que costaron tiempo)

**La regla `h ≤ 21k` documenta la intención, no el corte real.**
Medido: a `h=64` todavía sobreviven **2** rezagos, no 1. Los meses tienen 19-23
ruedas y el lookup usa aritmética de mes calendario. Esto rompió **dos**
verificaciones escritas contra la regla en vez de contra el dato.
→ Cualquier chequeo que dependa de cuántos rezagos hay tiene que leer el
contador que `_build_lag_posicion_mes` devuelve.

**pyarrow imprime `(tabla, objetivo)`, al revés de como se lee.**
El mensaje nativo de `Table.cast` lista primero la tabla y después el esquema
objetivo, sin decir cuál es cuál. Me llevó a diagnosticar al revés. Por eso el
escritor ahora compara antes y reporta qué falta y qué sobra, con el nombre del
banco.

**`py_compile` no detecta `NameError` de runtime.**
Un refactor dejó `_cols_traer` referenciado en una línea de log después de
eliminarlo. Compiló perfecto y reventó a mitad de corrida, con minutos de carga
de datos ya gastados. → `pyflakes` sí lo ve; es el paso 0 del checklist.

**Convención invertida en `FEATURES_EXCLUIR`:**
una entrada **comentada significa feature ACTIVA**. Comentarla para "sacarla"
hace lo contrario. Así fue como `frec_flujo_pos` seguía entrando a la matriz.

**El diccionario de features publicado está desactualizado.**
`is_cierre_encaje` y `es_mes_cierre_trim` figuran como activas pero están
excluidas (línea 288 de v2). El conteo de 25 activas ya no es correcto. Si se
usa como checklist de revisión, esa parte desvía.

**Spyder corre con `runfile()` sin argumentos.** Los checklists tenían
`argparse` con `required=True` y abortaban. Ahora sin argumentos caen al
autotest sintético.

---

## 6. Próximo paso inmediato

```bash
cd "H:/DPINV/CARPETAS PERSONALES/DIEGO/3. Sistema Inteligente"
git add aux_verificar_ccovn_particion.py
git commit -m "Actualiza la asercion de contraparte de SISTEMA en el checklist CCOVN"
git push -u origin claude/build-feature-matrix-HDMPg
```

Y acto seguido, **la acción que realmente importa**: correr v2 completo, que
nunca terminó.

```
runfile('step001_build_feature_matrix_v2.py')
```

Con `particion_activa = "bbva"` (ya configurado). Falló tres veces por esquema
y los tres bugs están arreglados, pero **nadie confirmó todavía que llegue al
final**. Si vuelve a fallar por esquema, el mensaje ahora dice el banco y las
columnas exactas que faltan o sobran.

Cuando termine, validar en este orden:

```
runfile('aux_verificar_particion.py')            # --matriz para la real
runfile('aux_exportar_matriz_excel_v2.py')       # hoja CONCILIACION
```

La hoja `CONCILIACION` es el check que decide todo: `SISTEMA == FOCO + RESTO`
en el target, fila por fila. Si no cierra, los dos modelos no están partiendo el
mismo objeto que el agregado y compararlos no significa nada.

---

## 7. Comandos verificados en esta sesión

```bash
# Los tres checklists — TODOS PASAN al cierre
python aux_verificar_pos_ventana.py            # sin args = autotest sintético
python aux_verificar_particion.py
python aux_verificar_ccovn_particion.py

# Contra la matriz real (cuando exista)
python aux_verificar_pos_ventana.py --matriz "1. Data/Clean/matriz_features.parquet"
python aux_verificar_particion.py  --matriz "1. Data/Clean/matriz_features.parquet"

# Chequeo estático — 0 undefined en ambos
python -m pyflakes step001_build_feature_matrix.py
python -m pyflakes step001_build_feature_matrix_v2.py

# Dependencias que hubo que instalar
pip install pyflakes openpyxl
```

Config activa en v2 al cerrar:
`particion_activa="bbva"` · `LAGS_POSICION_MES=[1..8]` ·
`N_REZAGOS_OBJETIVO=4` · `POLITICA_ENCAJE_BBVA="exacto_y_sistema"`

---

## 8. Qué NO hacer

**No sumar cuantiles para recombinar FOCO + RESTO.**
`q_τ(A+B) ≠ q_τ(A) + q_τ(B)`. Sumar cuantiles supone comonotonía, y acá la
dependencia es probablemente **negativa** (cuando BBVA retira, otros bancos
aportan). No solo daría un intervalo demasiado ancho: iría en dirección
contraria al mecanismo que motiva la partición. Hay que simular trayectorias
conjuntas, sumarlas trayectoria por trayectoria, y recién ahí tomar cuantiles.

**El mismo error, ya detectado en el panel 3 del fan chart.** La mediana
acumulada deriva a −6.500 mientras el realizado termina cerca de 0. Si ese panel
acumula los cuantiles diarios, tiene dos errores compuestos: suma de medianas ≠
mediana de la suma (centro sesgado) y suma de cuantiles ≠ cuantil de la suma
(banda demasiado ancha). **Verificar antes de construir el MCO encima**, porque
el MCO hereda la deriva completa.

**No crear una familia paralela de features para comparar contra la original.**
Dos columnas correlacionadas se reparten la importancia en permutación y las dos
salen deprimidas. Comparar entre corridas, no dentro de una.

**No confiar en el conteo de cobertura para validar un emparejamiento.**
`CREDITO` mapeaba a `COOPERATIVA DE AHORRO Y CREDITO ABA` y **contaba como
emparejado**: la cobertura del 86% se veía sana mientras el banco más grande del
resto apuntaba al saldo de una cooperativa. El conteo detecta un match ausente,
nunca uno equivocado. Hay que verificar contra el resultado esperado.

**No escribir verificaciones contra una regla documentada** cuando existe el
dato real. Ver el gotcha de `h ≤ 21k`.

**No agregar `n_lags_pos` como feature.** Ya existe (`GUARDAR_N_LAGS_POS`) y
está correctamente desactivada: dentro de un modelo de h fijo es casi constante,
así que solo agrega candidatos de corte inútiles.

---

## 9. Pendientes menores anotados

- `BONY` y `FEDERAL` no existen en `Saldos_CCOVN.xlsx`. Degradan a NaN, que es
  correcto: no son globales y el resto se calcula por diferencia. Documentado,
  no requiere acción.
- Máscara point-in-time de `_build_lag_posicion_mes` es `f_lag <= ft`,
  **inclusive**: en algunas filas lee `R`/`D` de `fecha_t` mismo, mientras que
  `R_t0`/`D_t0` están excluidos de la matriz por no ser información de decisión.
  Los dos criterios no coinciden. **No se cambió** porque alteraría los valores
  de toda la familia e invalidaría las mediciones ya hechas. Decisión pendiente.
- Completar la grilla ancla × componente de `*_pos`: falta
  `esc_neto_min_pos_ap` (**sale gratis**, ya se calcula y se descarta en el
  `_` de la llamada del bloque `_ap`) y `esc_retiro_pos_ap` (tiene hipótesis
  propia: la reversión depósito→retiro al abrir el mes está anclada a la
  apertura, no al cierre). Leer primero el resultado de la ventana deslizante:
  si el decaimiento en `h` se aplana, la familia gana peso.
- Features de intervalo `[t, t+h]` (cuántos cierres contiene, posición del
  primero): es la mayor ganancia disponible según los heatmaps de permutación,
  donde todo lo anclado a fecha domina y todo lo `_lag1` queda al fondo.
