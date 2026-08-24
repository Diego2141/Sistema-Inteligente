# Handoff — Build feature matrix for liquidity prediction

**Sesión:** 2026-08-24
**Branch:** `claude/build-feature-matrix-HDMPg`
**HEAD al cerrar:** ver `git log -1` (último commit de la sesión: handoff + alineación de `FAMILIAS_PERM`)
**Working tree:** limpio, todo pusheado a `origin`

> No existía `HANDOFF_TEMPLATE.md` en el repo. Este archivo usa la estructura
> pedida en la instrucción.

---

## 0. Contexto del proyecto (leer primero si venís sin contexto)

**Qué es.** Sistema de predicción de liquidez en moneda extranjera del BCRP
(banco central peruano). Se modela el **flujo neto diario** `D − R` (depósitos
menos retiros) del sistema bancario contra sus cuentas en el BCRP, con
**regresión cuantílica XGBoost, un modelo por horizonte**, h = 2..75 días
hábiles, τ ∈ {0.01, 0.05, 0.40, 0.50, 0.60, 0.95, 0.99}.

**Para qué.** El destino es la **liquidez operativa requerida**: cuánta caja
tiene que poder producir el tramo corto del portafolio de las RIN a cada plazo.
El objeto formal es el **MCO** (*maximum cumulative outflow*): la caída
acumulada máxima **desde el origen** a un percentil dado. No el flujo del día de
vencimiento, no el acumulado al horizonte: el punto más hondo del camino.

**Los tres hallazgos** que sostienen todo el diseño de features:

1. **Cuánto** — el cierre pasó de estar equilibrado (2010-2012) a una salida
   neta sostenida. Cambio de régimen alrededor de 2018-19, no intensificación.
2. **Cómo** — no se reparte parejo en el mes: el sistema **deposita al abrir** y
   **retira contra el cierre**, con pico más pronunciado en cierre de trimestre.
   *Es el hallazgo que justifica toda la familia `*_pos` y las dos anclas.*
3. **Quién** — **BBVA concentra ~94%** del neto del sistema en la ventana de
   cierre (2025), cinco años seguidos sobre 75% desde 2022. El segundo es
   Citibank con 9%. *Es lo que motiva las particiones.*

**Ventana de cierre** = últimos 5 días hábiles del mes. Misma definición en los
tres hallazgos.

**Las dos colas se anclan a extremos opuestos del mes**, y aparece en los
heatmaps sin que nadie lo impusiera: `dias_al_cierre_mes` domina q01 (retiros),
`dias_desde_cierre_mes` domina q99 (depósitos). De ahí el sufijo `_ap`
(ancla apertura) para las features orientadas a la cola alta.

### Pipeline

```
step001_build_feature_matrix[_v2].py   matriz de features -> Parquet
        |
step005_walk_forward_cv_4.py           XGBoost por horizonte + walk-forward CV
        |                              (de acá salen los heatmaps Block PERM)
        +--> aux_fanchart_cv4_direct.py    fan charts
        +--> step006_simulacion_paths_v2.py  simulación de trayectorias
        +--> step006_cqr_calibration.py      calibración conforme
```

Hay muchas versiones de cada step en el repo. **Las vigentes son
`step001_..._v2.py` (o v1 si no se usa partición) y `step005_walk_forward_cv_4.py`.**

### Rutas (Windows, unidad de red H:)

```
BASE = H:\DPINV\CARPETAS PERSONALES\DIEGO\3. Sistema Inteligente\
  1. Data\Raw\Transacciones_BancaLocal.xlsx    flujos D/R por banco (fuente principal)
  1. Data\Raw\Saldos_CCOVN.xlsx                saldos CC+OVN en BCR, 86 columnas
  1. Data\Raw\EncajeD.xlsx                     encaje diario
  2. Output\encaje_bbva\bbva_encaje_features_modelo.xlsx
  1. Data\Clean\matriz_features.parquet             SALIDA de step001 v1, ENTRADA de step005
  1. Data\Clean\matriz_features_particiones.parquet  SALIDA de step001 v2 (con FOCO/RESTO)
```

### Entorno

- **Se corre desde Spyder con `runfile()`**, sin argumentos de línea de comandos.
  Los scripts de verificación caen al autotest sintético si no reciben args.
- `step001` **pide credenciales de proxy en el import** si `BCRP_PROXY` no está
  en el entorno. Los checklists la declaran con un centinela para saltarlo.
- Las corridas de step001 tardan minutos (descarga series externas + procesa
  ~4.300 fechas × 74 horizontes × 9 entidades).

### Artifacts publicados (deliverables vivos)

- **One-pager de la reunión** — `onepager_encaje.html` en el repo →
  https://claude.ai/code/artifact/322e9f5b-c7e4-4fca-b3a7-b2e142d077a5
- **Diccionario de features** — `diccionario_features.html` en el repo →
  https://claude.ai/code/artifact/b8e796f5-693e-48f5-b473-99c4b65722cc

Para actualizarlos hay que republicar **con la misma URL**, si no se crea un
artifact nuevo y el link compartido queda viejo.

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

Nada a medio hacer en el código. Todo lo tocado está commiteado y pusheado, y
los tres checklists pasan. Lo que falta es **ejecutar**, no escribir: ver
"Bloqueado" abajo y §6.

### Bloqueado / no terminado

- **La corrida completa de v2 nunca llegó al final.** Falló tres veces seguidas
  por incompatibilidad de esquema entre bancos. Los tres bugs están arreglados,
  pero **la última corrida del usuario fue antes del último fix**, así que
  todavía no hay confirmación de que v2 termine y escriba el Parquet.
- **Recombinación FOCO+RESTO → total:** no implementada. Es el paso conceptual
  siguiente y tiene una trampa importante documentada en §8 ("Qué NO hacer"),
  que además **ya está presente en el código del fan chart**.
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

### `step005_walk_forward_cv_4.py` — **el entrenamiento XGBoost por horizonte**
Es el consumidor principal de la matriz: un modelo por horizonte, walk-forward
CV, y de acá salen los heatmaps de importancia por permutación en bloque
(`Block PERM (VAL/OOS)`) que guiaron todas las decisiones de features de esta
sesión.

**Editado en esta sesión** (único cambio): `FAMILIAS_PERM` (línea ~224).
v2 renombró las columnas CCOVN de naming absoluto (`ccovn_bbva_lag1`) a relativo
(`ccovn_propio_lag1` / `ccovn_contraparte_lag1`), así que la familia `ccovn_bcr`
se encogía **de 4 columnas a 2** y las tres nuevas no aparecían en ningún
heatmap. Degradación silenciosa: la nota de la línea 222 dice que los nombres
que no coinciden *"se ignoran en tiempo de ejecución, no truenan la corrida"*.
Ahora la familia lista los nombres de v1 **y** de v2 (9 columnas); los ausentes
se ignoran, así que sirve contra matrices de cualquiera de las dos versiones.

**Sin resolver:** `BANCO = "SISTEMA"` (línea 64) es una constante. Para entrenar
`FOCO_BBVA` y `RESTO_BBVA` hay que cambiarla y correr una vez por entidad, o
parametrizarla. Sin eso, la partición produce las filas en la matriz pero nadie
las entrena.

### `aux_fanchart_cv4_direct.py` — fan charts
`BANCO = "SISTEMA"` (línea 111), misma limitación que step005.

**Bug confirmado en el código, no arreglado** — ver §8.

### `onepager_encaje.html`, `diccionario_features.html`
Fuentes de los dos artifacts publicados, subidas al repo.

### Consumidores NO revisados en esta sesión
`step004_train_xgboost_qt*.py`, `step006_simulacion_paths_v2.py`,
`step006_cqr_calibration.py`, `step007_overlay_sobreencaje.py` y los demás
`aux_*`. **Ninguno se verificó contra el esquema de v2.** El renombre de CCOVN
y la exclusión de `frec_flujo_pos` pueden afectarlos igual que a step005.
Comando para auditarlos antes de usarlos está en §7.

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

**Recién después de eso**, entrenar con `step005_walk_forward_cv_4.py`. Ahí hay
que decidir algo que no está resuelto: `BANCO = "SISTEMA"` (línea 64) es una
constante, así que para entrenar `FOCO_BBVA` y `RESTO_BBVA` hay que correrlo una
vez por entidad cambiando esa línea, o parametrizarla. La partición ya produce
las filas en la matriz, pero sin ese cambio nadie las entrena.

Lo primero a mirar en el heatmap de la corrida nueva: **si el decaimiento en `h`
de la familia `*_pos` se aplanó**. Es la pregunta que la ventana deslizante
existe para responder: si se aplana, el decaimiento anterior era artefacto de
quedarse sin observaciones; si persiste, era señal real y conviene gastar el
presupuesto de columnas en features de intervalo en vez de en más miembros de
esa familia.

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
python -m pyflakes step005_walk_forward_cv_4.py

# Auditar qué consumidores aguas abajo referencian columnas que v2 ya no
# produce. Sirve para step004/006/007 y los aux_*, que NO se revisaron.
grep -rn "ccovn_bbva_lag1\|var_ccovn_bbva_lag1\|bbva_share_lag1\|frec_flujo_pos" \
     step0*.py aux_*.py

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

**El mismo error ya está EN EL CÓDIGO, confirmado.**
`aux_fanchart_cv4_direct.py:339-345`:

```python
cum_q01 = np.cumsum(res["q01"].values / 1e6)
cum_q50 = np.cumsum(res["q50"].values / 1e6)
cum_q40 = np.cumsum(res["q40"].values / 1e6)   # ... idem q60, q95, q99
```

El panel 3 acumula **cada trayectoria de cuantil por separado**. Dos errores
compuestos:

1. `cumsum(q50)` = suma de medianas ≠ mediana de la suma. Con una distribución
   asimétrica la mediana diaria es levemente negativa aunque la media sea cero,
   y ese sesgo se multiplica por 75. **Es la deriva a −6.500 observada** mientras
   el realizado acumulado termina cerca de 0.
2. `cumsum(q40)`/`cumsum(q60)` suponen que TODOS los días salen en el mismo
   percentil — probabilidad ~nula sobre 75 días. La banda es mucho más ancha que
   el intervalo verdadero del acumulado.

No se puede arreglar desde los cuantiles marginales: hacen falta trayectorias.
Tres opciones, en orden de esfuerzo: (a) reetiquetar el panel como banda
comonotónica y advertir que sobreestima, (b) quitar las bandas acumuladas y
dejar solo el realizado, (c) alimentarlo desde `step006_simulacion_paths_v2.py`,
que ya existe en el repo y es la fuente correcta.

**Crítico para el producto:** el MCO es un máximo sobre el camino acumulado, así
que hereda la deriva entera. **Resolver esto antes de construir el MCO encima.**

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

## 9. Hilo paralelo: el one-pager de la reunión

Buena parte de la sesión fue redactar el one-pager para la reunión con el
**equipo de Operaciones Monetarias**. Está publicado y en el repo
(`onepager_encaje.html`). Estado: **terminado y publicado**, salvo la sección
que se estuvo iterando.

**Reencuadre pedido por el jefe, ya aplicado.** El proyecto NO debe presentarse
como una herramienta para la gestión de la mesa de liquidez, porque eso obliga a
trabajar en conjunto con esa área. Se enfoca en **identificar riesgos** y en
aportar a un **mejor portafolio de referencia**.

**Sección "Por qué importa" — versión final aprobada** (está en el artifact
publicado). Argumento, en orden:

1. El requerimiento no es el mismo según el día del mes desde el que se lo mire.
   Día hábil 3 contra día 18, mismo plazo, requerimiento distinto.
2. Los indicadores actuales son **históricos**; lo propuesto es **simulado y
   prospectivo**.
3. Segunda ganancia: durante el mes **también entra** liquidez, y hoy esa
   entrada no se cuenta.
4. Recuadro: reconocer la trayectoria completa **libera requerimiento al inicio
   del mes y lo endurece cerca del cierre**; el error cambia de signo con el
   calendario.
5. Al construirse sobre **percentiles**, entrega un rango y permite escenarios.
6. Si el patrón resulta estable, potencialmente se convierte en un **portafolio
   de referencia** cuya composición de plazos acompaña el ciclo del mes.

**Preferencias de redacción del usuario, que aplican a todo el documento:**
- **Sin guiones largos (—).** Prefiere comas.
- Las notas de tablas y gráficos van como **bullets**, no como párrafo.
- No mencionar explícitamente que se llama "sistema inteligente".
- No nombrar el indicador vigente (LaR) en el documento; describirlo
  genéricamente como "los indicadores con que se mide hoy".

**Dato de contexto que no está en el documento:** el indicador vigente es
**LaR sobre ventanas acumuladas históricas**. Las tres brechas identificadas
contra él: no distingue el punto del mes, mide el punto final de la ventana y no
el más hondo del camino, y mezcla los dos regímenes de la muestra.

---

## 10. Pendientes menores anotados

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
