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
`generar_video_fancharts.py` necesita además `imageio` + `imageio-ffmpeg`, y solo
esos dos: si falta el segundo, `imageio` cae en otro plugin y el error que emite
(`TiffWriter.write() got an unexpected keyword argument fps`) no dice nada sobre
la causa — por eso el script lo verifica antes de escribir.

**`multiprocessing` en Windows usa `spawn`, no `fork`.** Cada worker es un
`python.exe` nuevo que reimporta numpy/scipy/pandas desde cero: ~150-250 MB de
*commit charge* por worker. Con `N_JOBS = -1` en una máquina de 16 cores eso son
~4 GB extra, y si el archivo de paginación es chico Windows no crea el proceso y
tira `OSError: [WinError 1455] El archivo de paginación es demasiado pequeño`.
No es falta de RAM libre: el límite de commit es RAM + pagefile. Arreglo sin
permisos: `N_JOBS = 4`. Es el mismo orden de magnitud al que aterriza
`get_max_workers_optuna()` de `step005`.

## Pipeline

```
step001_build_feature_matrix.py        →  1. Data/Clean/matriz_features.parquet
step001_build_feature_matrix_v2.py     →  1. Data/Clean/matriz_features_particiones.parquet
        │                                  (formato largo: una fila por banco × fecha_t × h)
        ↓
step005_walk_forward_cv_4.py           →  XGBoost por horizonte, walk-forward CV
step005_walk_forward_cv_3.7.py         →  la que tiene PARTICIONES + HMM interno
        │                                  produce los heatmaps Block PERM de importancia
        │                                  y preds_test_fold*_<banco>_*.parquet
        ├→ step005_validar_hmm_v5.py       régimen HMM (la llama cv_3.7 solo)
        ├→ aux_fanchart_cv4_direct.py      fan charts
        ├→ step006_orquestador_vf_7.py     simulación de trayectorias → MCO
        │      └→ step006_simulacion_paths_vf7.py   (el motor que importa)
        │            └→ generar_video_fancharts.py  ensambla los PNG en video
        └→ step006_cqr_calibration.py      calibración conforme
```

### El objeto de decisión y sus dos etapas

`step005` produce **marginales**: `q01..q99` de `D−R` por `(banco, fecha_t, h)`.
`step006` produce la **conjunta**: acopla esas marginales con una cópula gaussiana
AR(1) y acumula. Los dos insumos de dependencia los estima `step005` en
VALIDACIÓN y viajan en `preds_test`:

| columna | qué es | quién la consume |
|---|---|---|
| `rho_s_0..rho_s_{n-1}` | `φ_i(s)` — autocorrelación temporal (D1) | la recursión AR(1) |
| `rho_ij`, `rho_ij_s*` | `ρ_ij(s)` — correlación entre grupos (D2) | solo el modo conjunto |
| `condicionar_por` | sobre qué está estratificado el subíndice `s` | los guards de step006 |

La metodología está en *"Simulación conjunta de flujos netos por grupos del
sistema financiero"*: `Σ_e = R ⊙ (11ᵀ − φφᵀ)` (ec. 5), cota cerrada para N=2
(ec. 7), encogimiento `R(λ) = λR + (1−λ)I` (ec. 8), algoritmo P1-P5. Las cuatro
funciones que lo implementan viven en `step006_orquestador_vf_7.py`, no en el
módulo de simulación, y son puras — se validan sin acceso a `H:`.

### Botones que cambian qué se estima o se simula

| archivo | botón | qué hace |
|---|---|---|
| `step005_..._3.7` | `PARTICIONES` | FOCO/RESTO en vez de SISTEMA |
| | `BANCO_REGIMEN` | de qué entidad sale el estado (`"SISTEMA"`) |
| | `HMM_INTERNO` | ajusta el HMM por su cuenta, alineado a los folds |
| | `CONDICIONAR_POR` | `"regimen"` (HMM) o `"calendario"` (4 baldes) |
| | `MODO_DEBUG` | corrida de ~10 min en vez de ~60 |
| `step006_orq_vf_7` | `PARTICIONES` | N=1 (una entidad) o N=2 (conjunta, P1-P5) |
| | `PARTICION` | `"bbva"` o `"globales"` |
| | `ENTIDAD` | `SISTEMA`/`FOCO`/`RESTO` — **solo se lee con `PARTICIONES=False`** |
| | `CONDICIONAR_POR` | `"auto"` (lo deduce de `preds_test`), `"regimen"`, `"calendario"` |

### Las 16 combinaciones válidas de `step006`

Los cuatro botones no son independientes —cada uno apaga la lectura de algún
otro— así que el producto cartesiano (36) confunde. Las familias reales son
cinco, y están tabuladas en la cabecera de `step006_orquestador_vf_7.py`:

| # | `PARTICIONES` | `PARTICION` | `ENTIDAD` | `CONDICIONAR_POR` | `BANCO` | N | fan charts |
|---|---|---|---|---|---|---|---|
| 1 | False | (inerte) | SISTEMA | auto \| regimen | `SISTEMA` | 1 | los tres |
| 2 | False | bbva\|globales | FOCO | auto \| regimen | `FOCO_<P>` | 1 | los tres |
| 3 | False | bbva\|globales | RESTO | auto \| regimen | `RESTO_<P>` | 1 | los tres |
| 4 | True | bbva\|globales | (ignorado) | auto \| regimen | `CONJUNTO_<P>` | 2 | solo acumulado |
| 5 | True | bbva\|globales | (ignorado) | calendario | `CONJUNTO_<P>` | 2 | solo acumulado |
| X | False | * | * | **calendario** | — | — | **aborta al importar** |

Cuentan 5 entidades N=1 × 2 modos = 10, más 2 particiones × 3 modos = 6.

Las tres reglas que explican la tabla:

- **`PARTICION` solo se lee cuando hay partición de la cual hablar**: con
  `PARTICIONES=True` siempre, y con `False` solo si `ENTIDAD` es FOCO o RESTO.
  Con SISTEMA su valor no toca ninguna ruta ni ningún nombre — por eso un typo
  ahí **no** aborta, sería ruido sobre un campo inerte.
- **`ENTIDAD` solo se lee con `PARTICIONES=False`.** El modo conjunto corre las
  dos caras por definición: `ρ_ij` entra una sola vez, en el `η_h` vectorial de
  P2. Correr dos veces y sumar da `ρ_ij = 0` efectivo.
- **`calendario` corre en la ruta CONJUNTA y no en la N=1.** `main_conjunto`
  arma la secuencia de baldes y se la pasa a `secuencia_regimen(baldes_fijos=)`;
  la ruta N=1 delega en `pipeline_simulacion` de vf7, que llama a
  `simular_regimen_path` por dentro y no expone dónde inyectarla. Soportarlo
  exige tocar el motor vigente.

La fila X se rechaza **al importar** (`_validar_combinacion`). Antes el rechazo
ocurría dentro del bucle de `año_corte`: una hora de cómputo para llegar a un
error que la config ya permitía anticipar.

`PARTICIONES=True` genera **solo el fan chart acumulado**, no los tres. No es
una limitación pendiente: `neto` e `integrado` leen percentiles de la marginal de
**una** entidad, y la del agregado no existe en forma cerrada — hay que
simularla, que es exactamente lo que hace el acumulado.

### El nombre del archivo lleva la configuración

`sufijo_config()` — definida igual en `step006_orquestador_vf_7.py` y en
`generar_video_fancharts.py`, y las dos se mueven juntas:

```
CONJUNTO_BBVA_1_0.5_condregimen     ← BANCO + geometría del fold + modo
```

Va en los 9 `.parquet` + el `.json` de `step006` y en el `.mp4` del video.
Las carpetas ya separaban por corrida (`_SUF_SALIDA`), pero eso solo protege
mientras el archivo **queda donde se escribió**, y los `.parquet` son justamente
los que uno copia afuera para comparar dos corridas. Además hay un caso en que
la carpeta no alcanza ni en su sitio: con `CONDICIONAR_POR="auto"` el modo no se
conoce al armar las rutas, así que `_SUF_SALIDA` no puede llevar el
`cond_<modo>` y una corrida de calendario pisaba a una de régimen.

Los **PNG de frames no llevan el sufijo**, a propósito: los nombra
`step006_simulacion_paths_vf7.py` como `fanchart_<banco>_<fecha>.png`, se
direccionan en bloque por carpeta y nunca salen de ella. Renombrarlos
invalidaría los frames ya generados sin resolver nada.

### Convención de versiones: hay muchas, importa cuál

El repo acumula versiones de cada step (`step005_walk_forward_cv.py`, `_2`, `_3`,
`_3.2_revision`, `_4`, `_5`, ...). **No asumir que el número más alto es el
vigente.** Las versiones en uso son:

- **`step001_build_feature_matrix_v2.py`** — agrega particiones del sistema.
  Usar `step001_build_feature_matrix.py` (v1) solo si no se necesita partición.
- **`step005_walk_forward_cv_4.py`** — es la que referencian los `aux_*` activos
  (`aux_fanchart_cv4_direct.py`, `aux_comparar_cv4_configs.py`,
  `aux_importancia_calendario.py`, entre otros).
- **`step006_orquestador_vf_7.py`** + **`step006_simulacion_paths_vf7.py`** —
  el orquestador importa el módulo de simulación; ojo que la numeración de los
  dos archivos **no está alineada** entre sí y estuvo desfasada un tiempo (el
  orquestador `vf_7` importaba `vf6`).
- **`step005_validar_hmm_v5.py`** — la llama internamente
  `step005_walk_forward_cv_3.7.py` con `HMM_INTERNO=True`.

Para confirmar cuál está vigente: `grep -l "cv4\|cv_4" aux_*.py`, y para el
módulo de simulación `grep -n "simulacion_paths" step006_orquestador_vf_7.py`.

**El `grep` es necesario pero no suficiente: también hay que mirar lo que NO
está commiteado.** `step006_simulacion_paths_vf7.py` existió sin versionar
mientras el orquestador importaba `vf6`, así que el grep devolvía `vf6` y era
la respuesta correcta a la pregunta equivocada. Antes de concluir que una
versión no existe:

```bash
git status --short          # los untracked son parte del inventario
git ls-files | grep <step>  # lo que sí está versionado
```

Y para comparar dos versiones sin leer miles de líneas, diffear la **superficie**
(clases, funciones y constantes de nivel superior) con un recorrido del AST, no
el texto completo: `vf6` y `vf7` difieren en 239 líneas, y el AST muestra en dos
segundos que son 3 funciones y 2 constantes nuevas, 5 funciones reescritas, y que
todo el motor de simulación es idéntico.

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
| `aux_verificar_estructura_paths.py` | esquema de los `simulacion_paths_*.parquet` entre corridas |

`aux_verificar_estructura_paths.py` compara dos entregables de `step006` para
poder afirmar que sirven como el mismo insumo aguas abajo. Vale la pena porque
las dos ramas del orquestador arman ese parquet **por separado** —`main_conjunto`
con su propio `resultados.append`, y `main` (N=1) vía `pipeline_simulacion` de
vf7— y nada ata los dos diccionarios. Juzga como estructura las columnas, su
orden, la familia de dtype, la clave `(fecha_t, ventana, tau)`, los nulos y la
grilla de `tau`; informa sin fallar el rango de fechas y la grilla de `ventana`,
que son perillas de configuración y no contrato.

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

**Verificar que `pyflakes` esté instalado antes de confiar en su salida.** No
está en todos los entornos (`pip install pyflakes`). Si no está, `python -m
pyflakes` falla con `No module named pyflakes` y cualquier `grep -c "undefined
name"` sobre esa salida devuelve 0 — indistinguible de "pasó". Un `|| echo OK`
como fallback lo vuelve peor. Confirmar con `python -m pyflakes --version`
primero. Esto ya causó una sesión entera de verificaciones reportadas como
hechas que nunca corrieron.

**Ejercitar la función de producción con un *stub* en vez de reimplementarla.**
Para verificar la recursión latente de `step006` (V1 y V2 del paper: que
`Var(Z)=1` y que se recuperen `φ_i` y `ρ_ij`) hay que inspeccionar los `Z`, que
la función no devuelve. La solución es inyectar una marginal identidad
(`ppf = Φ⁻¹`), con lo que `X == Z` exactamente y se audita el espacio latente
**a través** del código real, sin copiar la recursión al test.

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

**El MCO todavía NO se calcula en ninguna parte.** Es el objeto formal del
sistema según la primera sección de este archivo, pero `step006` acumula
`flujos[h <= v].sum()` — el acumulado **al** horizonte, no el mínimo del camino.
No hay `np.minimum.accumulate` ni equivalente en `step006_simulacion_paths_vf7.py`
ni en el orquestador. Medido sobre paths AR(1) con drift: el MCO es **~5% más
hondo** que el acumulado al horizonte.

Y ojo con el atajo: `VENTANAS = [2..75]` cubre todos los horizontes, así que
tienta tomar el mínimo sobre `ventana` de los percentiles ya calculados. **Es el
mismo error de no-aditividad en otro disfraz** — `min_v q_τ(acum_v) ≠
q_τ(min_v acum_v)`; medido, subestima 5%. El MCO hay que calcularlo **dentro**
del bucle de réplicas, no derivarlo después.

**"Conservador" en un AR(1) es el φ más grande ALGEBRAICAMENTE, no el de mayor
magnitud.** `Var(Σ_{h=1..H} Z_h)` es monótona creciente en φ sobre todo `(-1, 1)`:
a H=75 la sd va de 1.56 en φ=−0.95 a 46.70 en φ=+0.95. Con φ<0 más magnitud da
**menos** dispersión del acumulado, o sea menos requerimiento de liquidez. El
fallback de `_estimar_rho_val_fold` tuvo el bug espejo: filtraba a los φ
positivos con el comentario "la persistencia genuina es positiva" y caía a un
piso de +0.3 cuando todos los estimados eran negativos — que es el caso real
(ver contexto de negocio). Hoy la jerarquía es: max algebraico de los estimados
→ φ pooled sin estratificar → `rho_default`.

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

### Cuarto hallazgo: el grupo FOCO revierte, el RESTO no

Medido sobre las dos particiones, con φ estimado en VAL por fold:

| | FOCO | RESTO |
|---|---|---|
| GLOBALES, moderado | **−0.373 ± 0.081** (negativo en 4/4 folds) | +0.002 |
| GLOBALES, severo | −0.230 (4/4 negativo) | −0.051 |
| BBVA, moderado | **−0.416** | +0.010 |
| BBVA, severo | −0.329 | +0.197 |

`φ_FOCO(moderado)` es el parámetro más estable de toda la tabla. No es ruido:
es **reversión a la media**, tesorería de ida y vuelta (deposita un día, retira
al siguiente). El grupo RESTO es ~ruido blanco a rezago 1.

Verificado que no es artefacto de `z = flujo/σ_EWMA`: con flujos iid simulados
(normal, t₃, y con 10% de ceros), `φ(z)` sale en `−0.003 ± 0.023`, y una
reversión real de −0.35 se recupera como −0.32. La normalización **no** fabrica
autocorrelación; si acaso atenúa ~7%.

Consecuencia directa sobre el requerimiento: la reversión **comprime** el
acumulado. A H=63, sd 5.54 contra 7.94 de una serie iid — **30% menos**. Quien
sume cuantiles día a día está sobreestimando fuerte el requerimiento de FOCO.

**`ρ_ij` es negativa y estable**: −0.0413 ± 0.0138 en GLOBALES, negativa en los
4 folds. Los dos grupos se **compensan**: agregar reduce el requerimiento frente
a sumar las partes. Es el resultado opuesto al de BBVA en la ventana de cierre y
vale decirlo explícito al reportar.

**Condicionar `ρ_ij` por régimen HMM agrega ruido, no señal** (GLOBALES):
`|sd/media|` = 0.33 en el global contra 1.22, 2.28 y 36.3 en los condicionales,
y los tres cambian de signo entre folds mientras el global nunca. Es lo que
motivó el botón `CONDICIONAR_POR`.

### Coverage por debajo del nominal — el problema abierto que domina

Contra un 90% nominal:

| entidad | coverage | implica |
|---|---|---|
| FOCO_GLOBALES | 88.8% (bajando 91→86 por fold) | intervalos 3% angostos |
| RESTO_GLOBALES | **83.0%** (peor fold 76.8%) | intervalos **20%** angostos |

Con sesgo VAL−TEST de +3 a +6 pp: Optuna elige hiperparámetros que rinden en VAL
y no transfieren. Importa **direccionalmente**: la skew-t de `step006` se ajusta
a esos cuantiles, así que cada día simulado sale de una distribución 20% angosta
y el requerimiento sale corto. Es el sentido equivocado del error.

Puesto al lado de los otros sesgos medidos, todos hacia subestimar:

| fuente | magnitud |
|---|---|
| **coverage insuficiente (RESTO)** | **~20%** |
| MCO no calculado | ~5% |
| Pearson vs normal scores en la ρ de la cópula | ~5% |
| atenuación EWMA | ~7% |

El primero domina por un factor de 4, y `step006_cqr_calibration.py` es la
herramienta que le corresponde. Los tres estadísticos nuevos de
`step006_simulacion_paths_vf7.py` (Wilson, Newey-West, Anderson-Darling sobre el
PIT) son los que diagnostican si la brecha es significativa.

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
