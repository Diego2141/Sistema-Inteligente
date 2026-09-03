# Handoff — Agregación por grupos (metodología de cópula gaussiana)

**Sesión:** 2026-08-31 a 2026-09-03 (varias sesiones consecutivas, mismo hilo)
**Branch:** `claude/build-feature-matrix-HDMPg`
**HEAD al cerrar:** `d5d7e25`
**Working tree:** limpio, todo pusheado a `origin`

> Reemplaza el handoff anterior (2026-08-24, particiones/feature engineering).
> Ese trabajo ya está mergeado y cerrado. Este documento es del hilo vigente:
> implementar la metodología de agregación de flujos por grupos (documento
> "Simulación conjunta de flujos netos por grupos del sistema financiero"),
> que extiende la simulación de paths (hoy escalar, N=1) a un caso conjunto
> con cópula gaussiana entre FOCO y RESTO de una partición.

---

## 0. Contexto que cambió desde el handoff anterior

**La cadena de scripts vigente ya NO es la que decía el handoff de agosto.**
Verificado con `git log --diff-filter=A` en todo el historial: estos 4 archivos
**nunca estuvieron en el repo** hasta esta sesión — vivían solo en `H:\` del
usuario y se subieron a pedido:

```
step005_validar_hmm_v5.py       ajusta el HMM de régimen (GaussianHMM 3 estados)
step005_walk_forward_cv_3.7.py  el walk-forward vigente (NO cv_4.py — ver abajo)
step006_orquestador_vf_7.py     orquesta la simulación de paths
step006_simulacion_paths_vf6.py motor de simulación (AR(1) + PIT + marginal)
```

`step005_walk_forward_cv_4.py`, que el handoff de agosto marcaba como vigente,
**no tiene** la maquinaria de régimen HMM ni de `rho_s`/`rho_ij` — toda esa
lógica vive en la línea `cv_3.x`. Si retomás este hilo, trabajá sobre
`cv_3.7.py`, no sobre `cv_4.py`.

**Patrón de nombres engañoso, ya verificado varias veces:** el código y los
comentarios referencian `validar_hmm_v3`, `v6`, o `step005_walk_forward_cv_4.py`
para cosas que en realidad vive en `v5`/`cv_3.7`. No confiar en el nombre que
aparece en un comentario — confirmar con `grep`/`git log` cuál archivo importa
cuál.

---

## 1. Qué se está implementando — el problema de fondo

**Objeto de negocio:** el MCO (*maximum cumulative outflow*) — la caída
acumulada máxima al percentil 5%, sobre una ventana de `h0..h` días. Hoy se
simula **una sola serie** (SISTEMA o una entidad). El paper pide simular
**FOCO y RESTO conjuntamente**, para poder sumar sus caminos simulados día por
día en vez de sumar sus percentiles por separado (error de comonotonía, ver
§8 del handoff de agosto — sigue vigente y sin resolver en
`aux_fanchart_cv4_direct.py`).

**La pieza que faltaba para eso:** `rho_ij`, la correlación entre FOCO y RESTO,
condicionada por régimen. Sin ella no se puede construir `Σ_e(s) = R(s) ⊙
(11' − φφ')`, la matriz de innovaciones de la ecuación (5) del paper.

**Fórmula general** (confirmada y usada en la implementación):
```
G grupos, S regímenes
→ φ:     G × S   valores   (autocorrelación temporal, uno por entidad×régimen)
→ ρ_ij:  C(G,2) × S valores (correlación cruzada, uno por par×régimen)
```
Con G=2 (FOCO, RESTO), S=3: 6 φ + 3 ρ_ij = 9 números por `Σ_e(s)`.

---

## 2. Estado — hecho y verificado

### El bloqueo que se resolvió: HMM degenerado en FOCO_BBVA

**Síntoma:** con `FOCO_BBVA` como entidad, `rho_ij` nunca aparecía en el
parquet de salida, sin ningún error — el pipeline terminaba "exitoso".

**Causa raíz, diagnosticada con evidencia (no supuesta):**
1. `FOCO_BBVA` tiene **10.0% de días con flujo exactamente cero** (vs 0.3% en
   SISTEMA) — es un solo banco que algunos días no opera; el agregado del
   sistema casi nunca da cero exacto.
2. Esos ceros vienen en **rachas de 1.4 días en promedio** (dispersos, no
   agrupados) — exactamente lo que rompe la persistencia (`diag(A)`) de un
   `GaussianHMM`: el estado que los cubre tiene que entrar y salir día por
   medio.
3. Resultado: `diag_ok=False` en **los 3 folds**, siempre — no es mala suerte
   de una corrida, ni de `HMM_N_STARTS=20` reintentos.
4. Había además un **bug de silencio total**: cuando `_fold_degenerado=True`,
   el bloque de `rho_s`/`rho_ij` se saltaba sin ningún `logger.warning`. Se
   corrigió (commit `9080ec7`) antes de encontrar la causa raíz real.

**La decisión que lo resuelve (tomada por el equipo, no unilateral):**
el **estado** del régimen (`regimen_hmm`) sale de **SISTEMA** para todas las
entidades — no cada una con el suyo. Está en línea con el paper: `Σ_e(s_h)`
usa un solo subíndice de régimen por horizonte, no uno por grupo. La **sigma**
(`regimen_sigma`) sigue siendo de cada entidad — es la escala que estandariza
`z_t = flujo_t/sigma_t` de esa entidad específica.

```python
# step005_walk_forward_cv_3.7.py:168
BANCO_REGIMEN = "SISTEMA"    # None = comportamiento anterior (por entidad)

# step006_orquestador_vf_7.py:80 — DEBE coincidir, es la misma decisión
BANCO_REGIMEN = "SISTEMA"
```

**Resultado confirmado con una corrida real** (FOCO_BBVA, MODO_DEBUG, 2026-09-03):
sin warning DEGENERADO, `rho_ij` calculado en los 3 folds:

| fold | año_corte | rho_ij global | rho_ij severo | n_pares |
|---|---|---|---|---|
| 1 | 2022-07-01 | (ver log) | — | — |
| 2 | 2023-01-01 | −0.019 | +0.039 | 848 |
| 3 | 2023-07-01 | +0.056 | +0.199 | 970 |

### Commits de esta sesión, en orden

```
53f1013  MODO_DEBUG — corrida de diagnóstico ~10 min en vez de ~60
9ad8ffc..4c2c21b..07c1071..cad157b  subida de los 4 archivos que faltaban en el repo
0e98eef  cv_3.7 genera el HMM por entidad, alineado a sus folds (asegurar_regimenes_hmm)
84cf779  _estimar_rho_transversal — cálculo de rho_ij, misma base que phi
56c083e  genera el régimen de la contraparte SIN correr su walk-forward completo
dcd9d52  fix: _Tee sobrevive a un corte de la unidad de red (crasheaba al final de corridas largas)
1a6ab8c  fix: validar_hmm_v5 no importaba en Python <3.10 (PEP 604 sin __future__.annotations)
9080ec7  fix: avisar cuando un fold se salta rho_s/rho_ij por HMM degenerado (antes: silencio total)
ba495a5  el ESTADO del régimen sale de SISTEMA para todas las entidades (la decisión del equipo)
d5d7e25  reporte consolidado CSV/Excel de phi_s y rho_ij por fold
```

### Verificación

Todo lo anterior se probó **ejercitando las funciones de producción reales**
vía `importlib` sobre parquets sintéticos (nunca reimplementaciones en
paralelo), con controles negativos y datos discriminantes (p.ej. escribir
`estado=9` en una fuente y `estado∈{0,1,2}` en otra para confirmar de dónde
sale cada columna). Los scripts de verificación se corrieron en consola y se
descartaron a pedido del usuario — no quedaron como archivos del repo, están
documentados en los mensajes de commit.

---

## 3. Pendiente — próximo paso concreto

### ① Correr con `ENTIDAD="RESTO"` para completar los 6 φ

Hoy solo se corrió `ENTIDAD="FOCO"`. Los `φ_RESTO(s)` (3 valores) faltan.
Config: `PARTICIONES=True`, `PARTICION="bbva"`, `ENTIDAD="RESTO"`,
`MODO_DEBUG=True`. Debería ser más rápido — el HMM de SISTEMA ya está
generado, `asegurar_regimenes_hmm` no reajusta nada.

**Chequeo de consistencia que da gratis:** `rho_ij` calculado desde FOCO debe
ser **idéntico** al calculado desde RESTO (mismo par, mismo régimen). Si
difiere, hay desalineamiento de fechas entre bloques HMM.

### ② Filtro de días inactivos (flujo=0) — deliberadamente NO incluido aún

El 10% de ceros de `FOCO_BBVA` ya no rompe el HMM (viene de SISTEMA), pero
**sigue** entrando en `z_t = flujo_t/sigma_t` cuando se calculan `φ` y `ρ_ij`
— atenuándolos hacia cero. Se decidió a propósito dejarlo fuera del cambio de
régimen para poder atribuir el efecto de cada cambio por separado (ver
pregunta del usuario y respuesta en el historial de conversación).

**Diseño propuesto, no implementado:** modelo de dos partes
(Stern & Coe 1984, "occurrence-amount"; ver Zucchini/MacDonald/Langrock 2016
para el tratamiento HMM de emisiones zero-inflated). Filtrar `flujo != 0` al
formar los pares en `_estimar_rho_val_fold` y `_estimar_rho_transversal`.

### ③ `Σ_e`, chequeo PSD, y λ* (bisección) — álgebra pura, sin datos

El bloque que construye `Σ_e = R ⊙ (11' − φφ')`, verifica
`λ_min(Σ_e) ≥ 0`, y si falla aplica el encogimiento `R(λ) = λR + (1−λ)I`
resuelto por bisección. **No depende de ninguna corrida** — se puede
implementar y testear con el ejemplo infactible del propio paper como control
negativo (N=3, φ=(0.90,0.50,0.10), ρ=(0.7,0.6,0.5) → λ*≈0.72).

Con la cota de factibilidad ya calculada en esta sesión para los φ reales
observados (todos con magnitud < 0.6), la cota da 0.96-1.00 — probablemente
λ* nunca se active en la práctica, pero el chequeo es obligatorio igual (el
paper lo llama explícitamente "comprobación obligatoria, no opcional").

### ④ Generalizar la recursión de `step006_simulacion_paths_vf6.py`

Hoy escalar (`z_k = ρ·z_{k-1} + √(1-ρ²)·w_k`, línea ~1139). La generalización
a N=2:
```python
Z[k] = Φ · Z[k-1] + L(s_k) @ η[k]      # η ~ N(0,I) vector 2D
                                        # L(s_k) = chol(Σ_e(s_k))
U = Φ_N(Z)                              # SIN cambios — sigue siendo norm.cdf componente a componente
X_i = F_i⁻¹(U_i)                        # SIN cambios — cada componente su propia marginal
```
La cópula gaussiana **ya está implementada** (`stats.norm.cdf(z)`, línea 1139
de `simulacion_paths_vf6.py`) — lo único que cambia es que el vector que entra
ahí pase de 1 a N componentes.

---

## 4. Decisiones no obvias de esta sesión

**Por qué el régimen es de SISTEMA y no por entidad — no es solo conveniencia
práctica.** El paper condiciona por `s_h` (un subíndice por horizonte), no
`s_{i,h}` (uno por grupo). `R(s_h)` es una correlación *entre* grupos; no
puede depender de un estado que cada grupo define por su cuenta, porque
entonces `φ_FOCO(2)` y `φ_RESTO(2)` se estimarían sobre conjuntos de días
DISTINTOS, y combinarlos en la misma `Σ_e(2)` no significaría nada.
Limitación aceptada: si FOCO y RESTO se compensan, SISTEMA puede verse
tranquilo mientras una entidad está tensionada — verificable comparando la
sigma de la entidad en días "calma" vs "severo" de SISTEMA.

**Por qué `regimen_sigma` NO sigue la misma regla.** Es la escala que
normaliza el flujo de *esa* entidad. Usar la sigma del agregado para
estandarizar el flujo de un banco individual mezclaría escalas.

**Por qué `asegurar_regimenes_hmm` genera SISTEMA/contraparte sin correr su
CV completa.** `rho_ij` no consume ninguna columna que el walk-forward de
XGBoost produzca — solo necesita `flujo`, `sigma`, `estado`, que salen del
ajuste del HMM (segundos), no del entrenamiento (minutos). Ahorra ~25 min por
entidad no evaluada.

**Por qué el modo sintético del `MODO_DEBUG` no toca el HMM.** Verificado
explícitamente: `MODO_DEBUG` solo reasigna `H_GRUPOS`, `TRIALS_FLAT`,
`DIAG_N_REPEATS`, `DIAG_SHAP_MAX_SAMPLES`, `DIAGNOSTICO_MEMORIA` — ninguna
toca `N_ESTADOS`, `HMM_MIN_DIAG_TRANSMAT`, `HMM_N_STARTS`, ni la serie de
flujo. El diagnóstico del HMM es válido en `MODO_DEBUG` tal cual.

**Por qué `regimen_hmm` no estaba llegando poblado a VAL/TEST antes de esta
sesión, y cómo se resolvió sin ese problema.** Cada bloque HMM se ajusta y
clasifica solo hasta su propio `train_end` — VAL/TEST quedan fuera del
bloque, y la columna se rellena con la **mediana de TRAIN** vía la imputación
estándar de `cols_feat` (no queda NaN). El propio autor ya lo sabía y por eso
`rho_s_val`/`rho_ij` se precomputan en step005 y se guardan en el parquet, en
vez de que step006 intente re-derivarlos sobre TEST con regímenes imputados.

---

## 5. Gotchas de esta sesión

**Un `if` sin `else` puede fallar en absoluto silencio.** El bug de
`_fold_degenerado` no logueaba nada en la rama `True` — costó una corrida
completa de 133 min para diagnosticarlo. Regla general aplicada después:
cualquier rama que decide "omitir esto" debe loguear por qué, aunque sea a
nivel `debug`.

**Variables sin inicializar sobreviven entre iteraciones de un `for`, sin
error.** `_fold_degenerado` no estaba definida fuera de su `if` — Python no
tiene scope de bloque, así que en un fold donde el `if` no se ejecutaba, la
variable conservaba el valor del fold **anterior** en vez de fallar con
`NameError`. Se corrigió inicializándola a `None` antes del `if` (`None`
≠ `False`: distingue "nunca se evaluó" de "se evaluó, no degenerado").

**`_Tee` (duplicador de log a archivo) crasheaba corridas largas.** Sobre
`H:\` (unidad de red), un corte momentáneo de conexión después de ~2h de
cómputo invalidaba el handle del archivo, y el `OSError` no atrapado mataba
el proceso **justo al final**, perdiendo solo el resumen de consola (los
parquets por fold ya estaban guardados). Se envolvió `write`/`flush`/`cerrar`
en `try/except OSError`.

**`list | None` (PEP 604) revienta en Python <3.10 si el módulo se importa
desde otro.** `validar_hmm_v5.py` nunca lo mostró por sí solo porque nunca se
importaba — el bug quedó invisible hasta que `cv_3.7` empezó a hacer
`import step005_validar_hmm_v5`. Arreglado con `from __future__ import
annotations` en la cabecera (mismo patrón que ya tenía `cv_3.7`).

**Retiros concentrados en cierre de mes por diseño regulatorio, no solo
tendencia estadística.** El propio código de `step001_build_feature_matrix_v2.py`
(línea ~3418) documenta: *"la reversión es sobre DEPÓSITOS... no sobre
retiros, que se concentran al cierre por diseño del encaje"*. Medido:
`eta²=11.2%` de la varianza del flujo de FOCO_BBVA se explica por
`dias_al_cierre_mes`, con medianas estrictamente monótonas
(−80M en el día de cierre vs −5M el resto del mes). Es evidencia de apoyo
para una eventual desestacionalización del HMM (no implementada, no
priorizada frente al filtro de ceros).

---

## 6. Comandos / snippets verificados en esta sesión

```python
# Confirmar que SISTEMA no está degenerado en ningún fold (ya corrido, sano en 8/8)
import pandas as pd
df = pd.read_parquet(r"H:\...\2. Output\estados_regimen_hmm_SISTEMA.parquet")
df.groupby("año_corte")["degenerado"].agg(["first", "all"])

# Ver el reporte consolidado de una entidad (NUEVO, commit d5d7e25)
base = r"H:\...\2. Output\step005_wfcv_v3\xgb_qt_expanding_310.5\FOCO_BBVA_1_0.5"
pd.read_csv(f"{base}\\rho_por_fold_FOCO_BBVA_<fecha>.csv")
```

Config verificada para reproducir la corrida que desbloqueó `rho_ij`:
```python
PARTICIONES = True
PARTICION   = "bbva"
ENTIDAD     = "FOCO"        # o "RESTO" para completar los φ que faltan
MODO_DEBUG  = True          # ~20 min en vez de ~2h; no afecta el HMM
BANCO_REGIMEN = "SISTEMA"   # ya es el default en el repo
```

---

## 7. Qué NO hacer

**No reintroducir régimen por entidad sin resolver antes el zero-inflation.**
Si algún día se decide volver a `BANCO_REGIMEN=None`, `FOCO_BBVA` va a
degenerar de nuevo en los 3 folds — es el mismo problema de datos, no algo
que el cambio de config por sí solo resuelva.

**No asumir que `step005_walk_forward_cv_4.py` es el vigente.** Es el error
que arrastraba el handoff anterior. La maquinaria de régimen/rho vive en
`cv_3.7.py`.

**No estimar `rho_ij`/`φ` sobre TEST.** Ya está resuelto (se precomputan en
VAL dentro de step005 y se persisten), pero si se toca ese código, no
reintroducir el antipatrón que el propio autor evitó a propósito.

**No sumar percentiles para combinar FOCO+RESTO.** Sigue vigente el error
documentado en el handoff de agosto (§8 de ese archivo, en el historial de
git si hace falta consultarlo) — `aux_fanchart_cv4_direct.py` todavía lo
tiene sin corregir.

**No dar por sentado que un HMM "no degenerado" en el log siempre lo dice.**
Verificar siempre contra la columna `degenerado` del parquet o el pickle
directamente — el log puede no mostrar la línea relevante si quedó fuera de
lo que se pegó/revisó.
