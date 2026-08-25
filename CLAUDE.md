# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Sistema de predicción de liquidez en moneda extranjera del BCRP (banco central
peruano). Modela el **flujo neto diario** `D − R` (depósitos menos retiros) del
sistema bancario contra sus cuentas en el BCRP, con **regresión cuantílica
XGBoost y un modelo por horizonte**: h = 2..75 días hábiles, τ ∈ {0.01, 0.05,
0.40, 0.50, 0.60, 0.95, 0.99}.

El destino del sistema es la **liquidez operativa requerida**: cuánta caja tiene
que poder producir el tramo corto del portafolio de las RIN a cada plazo. El
objeto formal es el **MCO** (*maximum cumulative outflow*): la caída acumulada
máxima **desde el origen**, a un percentil dado. No el flujo del día de
vencimiento ni el acumulado al horizonte, sino el punto más hondo del camino.

El código y los comentarios están en español. Mantenerlo así.

## Entorno y ejecución

**Los scripts se corren desde Spyder con `runfile()`, sin argumentos de línea de
comandos.** Cualquier script nuevo que use `argparse` debe funcionar sin
argumentos (por ejemplo, cayendo a un modo por defecto), o va a abortar al
correrlo desde el IDE.

`step001_build_feature_matrix*.py` **pide credenciales de proxy en el import** si
la variable de entorno `BCRP_PROXY` no está definida. Para importarlo desde un
script sin bloquear:

```python
os.environ.setdefault("BCRP_PROXY", "http://centinela-sin-red")
```

Los datos viven en una unidad de red Windows (`H:\...`) y **no están en el
repositorio** (ver `.gitignore`). Las rutas se configuran en el dict `PARAMS` de
`step001`. Sin acceso a `H:` no se puede correr el pipeline, solo los autotests
sintéticos.

No hay `requirements.txt`. Dependencias en uso: `pandas`, `numpy`, `xgboost`,
`pyarrow`, `matplotlib`, `scipy`, `statsmodels`, `openpyxl`, y opcionalmente
`hmmlearn`/`scikit-learn` (si faltan, `hmm_estado` queda en NaN sin romper).

## Pipeline

```
step001_build_feature_matrix.py        →  1. Data/Clean/matriz_features.parquet
step001_build_feature_matrix_v2.py     →  1. Data/Clean/matriz_features_particiones.parquet
        │                                  (formato largo: una fila por banco × fecha_t × h)
        ↓
step005_walk_forward_cv_4.py           →  XGBoost por horizonte, walk-forward CV
        │                                  produce los heatmaps Block PERM de importancia
        ├→ aux_fanchart_cv4_direct.py      fan charts
        ├→ step006_simulacion_paths_v2.py  simulación de trayectorias
        └→ step006_cqr_calibration.py      calibración conforme
```

### Convención de versiones: hay muchas, importa cuál

El repo acumula versiones de cada step (`step005_walk_forward_cv.py`, `_2`, `_3`,
`_3.2_revision`, `_4`, `_5`, ...). **No asumir que el número más alto es el
vigente.** Las versiones en uso son:

- **`step001_build_feature_matrix_v2.py`** — agrega particiones del sistema.
  Usar `step001_build_feature_matrix.py` (v1) solo si no se necesita partición.
- **`step005_walk_forward_cv_4.py`** — es la que referencian los `aux_*` activos
  (`aux_fanchart_cv4_direct.py`, `aux_comparar_cv4_configs.py`,
  `aux_importancia_calendario.py`, entre otros).

Para confirmar cuál está vigente: `grep -l "cv4\|cv_4" aux_*.py`.

## Arquitectura de `step001`: matriz de features

Produce **formato largo**: `fecha_t` (origen) × `h` (horizonte) × `banco`
(entidad). El target es `D(t+h) − R(t+h)`.

Distinción central que atraviesa todo el archivo:

- **`fecha_t`** — el origen. Features ancladas acá describen el estado conocido
  al decidir, con sufijo `_lag1`.
- **`fecha_th` = t+h** — la fecha objetivo. Features ancladas acá son de
  calendario o de posición en el mes, y son las que dominan la importancia:
  el calendario de la fecha objetivo es lo único que se conoce con certeza
  sobre el futuro, y su potencia **no decae con el horizonte**.

### La familia `*_pos`

Para cada `t+h` se busca el día de los meses previos que ocupa la **misma
posición respecto al cierre de mes** y se agrega sobre esos rezagos. Tres
sufijos conviven:

| Sufijo | Ancla | Sirve a |
|---|---|---|
| (ninguno) | cierre de mes | cola baja, retiros (q01) |
| `_ap` | apertura de mes | cola alta, depósitos (q99) |

Las dos anclas existen porque **las dos colas se anclan a extremos opuestos del
mes**, y eso aparece en los heatmaps sin que nadie lo imponga:
`dias_al_cierre_mes` domina q01, `dias_desde_cierre_mes` domina q99.

**Ventana deslizante de rezagos.** `LAGS_POSICION_MES` es una lista de
**candidatos** (`[1..8]`), no de rezagos que entran. De cada fila entran los
`N_REZAGOS_OBJETIVO = 4` más recientes que sobrevivan a la máscara point-in-time.
Antes entraban *todos* los disponibles de una lista de 4, y el conteo caía de 4 a
1 con el horizonte, dejando el feature de h largo como el valor de un único día.

**La regla `h ≤ 21k` documenta la intención, no el corte real.** Los meses tienen
19-23 ruedas y el lookup usa aritmética de mes calendario. Medido: a `h=64`
todavía sobreviven 2 rezagos, no 1. Cualquier verificación que dependa de cuántos
rezagos hay **tiene que leer el contador que `_build_lag_posicion_mes` devuelve**,
nunca derivarlo de un umbral de `h`.

### Particiones (solo en v2)

Parte el sistema en dos grupos complementarios que entran a `lista_bancos_full`
como dos "bancos" más, reusando toda la maquinaria de features:

```python
PARAMS["particion_activa"] = "bbva"      # FOCO_BBVA + RESTO_BBVA
                             "globales"  # FOCO_GLOBALES + RESTO_GLOBALES
                             None        # sin partición, idéntico a v1
```

**La partición se aplica sobre el pivot crudo, ANTES de `agrupar_bancos()`.** Es
obligatorio: esa función suma los bancos chicos en `Otros_bancos` y **elimina**
sus columnas, y varios bancos globales (Deutsche, ICBC, Bank of China, BCI) están
por debajo del umbral del 1%. Partir después los dejaría contados como banca
local sin ningún aviso. `aplicar_particion()` aborta con `RuntimeError` si
detecta que se la llamó tarde.

`FOCO + RESTO == SISTEMA` se verifica dos veces, sobre el pivot crudo y de nuevo
tras el agrupamiento, en vez de asumirse.

### CCOVN: roles relativos

Los saldos de cuenta corriente + overnight se resuelven **por entidad**:
`ccovn_propio_lag1` es el saldo del grupo modelado y `ccovn_contraparte_lag1` el
del otro lado de la partición. El naming relativo hace que la importancia por
permutación sea **comparable entre grupos**; con naming absoluto
(`ccovn_bbva_lag1`) la misma columna significaba "mi saldo" en un modelo y "el
del otro" en otro.

El emparejamiento entre `Transacciones_BancaLocal.xlsx` y `Saldos_CCOVN.xlsx`
usa `ALIAS_CCOVN` (tabla explícita) primero y subcadena normalizada después, con
`CCOVN_NO_BANCOS` como guarda: un banco nunca puede apuntar a una caja,
cooperativa, financiera o fondo. Sin esa guarda, `CREDITO` emparejaba con
`COOPERATIVA DE AHORRO Y CREDITO ABA` (en el archivo de saldos el Banco de
Crédito figura como `BCP`).

`resto = sistema − foco`, no suma de emparejados: solo los globales pueden
componer un FOCO, y `sistema` ya es la suma de todas las columnas sin depender
del matching.

## Invariantes que rompen la corrida si se violan

**Toda entidad debe producir el mismo juego de columnas.** El `ParquetWriter`
escribe banco por banco casteando contra el esquema del primero. Una rama
condicional que cree una columna para unas entidades y no para otras aborta la
corrida a mitad de camino. Reglas:

- Materializar en NaN lo que no aplica, en vez de omitir la columna.
- No crear los insumos intermedios que otra rama descarta (ver
  `BBVA_INTERMEDIAS`).
- El orden final de columnas es **alfabético**, no de inserción. En orden de
  inserción la posición depende del camino de ejecución y el esquema difiere
  entre entidades aunque el conjunto sea idéntico.

`aux_verificar_particion.py` audita esto estáticamente (sección 13): recorre el
AST de `build_feature_matrix` buscando columnas asignadas solo dentro de una rama
condicional y las contrasta contra sus materializaciones.

**`FEATURES_EXCLUIR` tiene la convención invertida:** una entrada **comentada
significa que el feature está ACTIVO** (no se lo excluye). Comentar una línea
para "sacar" un feature hace exactamente lo contrario.

Esa lista ya tuvo dos veces el bug de una coma faltante que concatena dos
literales adyacentes en uno solo (Python une strings pegados sin operador). Vale
verificarla con AST tras editarla.

## Verificación

**No hay pytest ni framework de tests.** La verificación son scripts
`aux_verificar_*.py` con checklists que imprimen `[OK]` / `[FALLA]` y salen con
código 1 si algo falla. Cada uno tiene dos modos:

```bash
python aux_verificar_particion.py               # autotest sintético, no necesita datos
python aux_verificar_particion.py --matriz "1. Data/Clean/matriz_features.parquet"
```

Checklists vigentes:

| Script | Cubre |
|---|---|
| `aux_verificar_pos_ventana.py` | ventana deslizante de rezagos, máscara point-in-time, recencia |
| `aux_verificar_particion.py` | particiones, mapeo CCOVN, paridad de esquema, orden determinista |
| `aux_verificar_ccovn_particion.py` | roles propio/contraparte por entidad |

(`aux_verificar_multioutput.py` no sigue este patrón: es un script de sondeo que
decide qué ruta hacia el modelo multi-output es viable con la versión de XGBoost
instalada, no un checklist de invariantes.)

Al escribir un checklist nuevo, la convención del repo es:

- **Ejercitar la función de producción**, no una copia de su lógica que pueda
  divergir. Si hace falta, extraer la lógica a una función y llamarla desde
  ambos lados.
- **Incluir controles negativos**: verificar que la comprobación falla cuando
  debe fallar. Un chequeo que nunca puede dar `[FALLA]` es decorativo.
- Series sintéticas construidas a propósito. Ejemplo útil: una serie cuyo valor
  es el ordinal de su propia fecha convierte "¿usó información futura?" en una
  comparación aritmética exacta.

**`py_compile` no detecta `NameError` de runtime.** Un refactor que elimina una
variable pero deja una referencia viva en una línea de log compila perfecto y
revienta a mitad de corrida. Usar:

```bash
python -m pyflakes step001_build_feature_matrix_v2.py   # debe dar 0 "undefined name"
```

Es el paso 0 de `aux_verificar_particion.py`.

## Artifacts publicados: leerlos por fragmento, nunca completos

Dos entregables vivos, con su fuente versionada en el repo:

| Fuente en el repo | Artifact publicado |
|---|---|
| `onepager_encaje.html` | https://claude.ai/code/artifact/322e9f5b-c7e4-4fca-b3a7-b2e142d077a5 |
| `diccionario_features.html` | https://claude.ai/code/artifact/b8e796f5-693e-48f5-b473-99c4b65722cc |

Para actualizarlos hay que republicar **con la misma URL**; sin ella se crea un
artifact nuevo y el link ya compartido queda apuntando a la versión vieja.

**`onepager_encaje.html` no se puede leer entero, y no hace falta.** Son 443 KB,
de los cuales **371 KB (84%) son una sola línea**: el `const DATOS = {...}` con
la serie diaria completa 2010-2026 embebida. Un `Read` del archivo aborta por
límite de tamaño, y un `WebFetch` del artifact publicado vuelca ese blob al
contexto. En una sesión eso costó más de 100k tokens sin aportar nada.

El contenido editable son ~70 KB repartidos alrededor de esa línea: el markup
antes y las funciones de gráficos después.

Flujo correcto para editarlo:

```bash
# 1. Ubicar la sección, sin traer el archivo
grep -n 'id="w3stat"\|const DATOS = ' onepager_encaje.html

# 2. Leer SOLO ese rango
#    Read con offset/limit alrededor de la línea encontrada
```

Y **saltear siempre la línea del `DATOS`**: se ubica con
`grep -n "const DATOS = "` y no se lee nunca. Es dato, no código.

Medido sobre este archivo: leerlo entero son ~113.000 tokens; el fragmento de 4
líneas que hace falta para editar una sección son ~130. **849 veces menos.**

La fuente de verdad es la copia del repo, no el artifact publicado: leer de local
en vez de hacer `WebFetch` evita el volcado inline por completo. El `WebFetch`
solo hace falta para recuperar una versión publicada que no esté en el repo, y
aun así guarda el HTML en un archivo local cuya ruta devuelve — hay que leer
fragmentos de **ese** archivo, no del resultado inline.

## Errores estadísticos a evitar en este dominio

**Nunca sumar ni acumular cuantiles.** `q_τ(A+B) ≠ q_τ(A) + q_τ(B)`, y
`cumsum(q_τ)` ≠ el cuantil del acumulado. Sumar cuantiles supone comonotonía:
que todos los componentes (o todos los días) salen en el mismo percentil a la
vez. Dos consecuencias concretas:

- La suma de medianas no es la mediana de la suma. Con una distribución
  asimétrica el sesgo diario se multiplica por el horizonte.
- La banda resultante es mucho más ancha que el intervalo verdadero.

Para recombinar componentes o acumular en el tiempo: **simular trayectorias
conjuntas que preserven la dependencia, sumarlas trayectoria por trayectoria, y
recién ahí tomar cuantiles.**

`aux_fanchart_cv4_direct.py:339-345` **tiene este error hoy** (`np.cumsum` sobre
cada trayectoria de cuantil, panel 3). Está documentado, no arreglado.

**Suma de sumas, no promedio de cocientes.** El promedio de participaciones
diarias no es la participación del período. Esta distinción ya causó confusión
al reportar la concentración de BBVA.

**Denominadores cerca de cero.** Cuando el neto del sistema se acerca a cero, las
proporciones explotan sin que el comportamiento haya cambiado. La convención del
repo es marcar esos casos (con `‡` o dejándolos vacíos) en vez de mostrar un
1.800% que se lee como dato en vez de como denominador chico.

## Contexto de negocio que explica el diseño

Tres hallazgos sostienen las decisiones de features:

1. **Cuánto** — el cierre de mes pasó de estar equilibrado (2010-2012) a una
   salida neta sostenida. Cambio de régimen alrededor de **2018-19**, no
   intensificación de algo previo.
2. **Cómo** — no se reparte parejo: el sistema **deposita al abrir el mes** y
   **retira contra el cierre**, con pico más pronunciado en cierre de trimestre.
   Justifica toda la familia `*_pos` y las dos anclas.
3. **Quién** — **BBVA concentra ~94%** del neto del sistema en la ventana de
   cierre (2025), cinco años seguidos sobre 75% desde 2022. Motiva las
   particiones.

**Ventana de cierre** = últimos 5 días hábiles del mes, misma definición en los
tres hallazgos.

Consecuencia metodológica: la serie tiene **dos quiebres de régimen** en 15 años
(2018-19 y 2022), así que la muestra efectiva del régimen vigente es corta. Toda
estimación de cola descansa sobre pocos años, y la cola está gobernada por la
política de tesorería de un solo banco, no por un agregado estadístico
diversificado.

## Git

Rama de trabajo: `claude/build-feature-matrix-HDMPg`. Los mensajes de commit del
repo son largos y explican **por qué**, no solo qué: qué se midió, qué
alternativa se descartó y con qué evidencia.

Hay handoffs de sesión en `handoffs/latest.md` con el estado verificado, el
próximo paso concreto y los gotchas acumulados. **Leerlo antes de retomar
trabajo.**
