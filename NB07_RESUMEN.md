# NB07 — Resumen para la defensa

**Notebook:** `notebooks/NB07_optimizacion_modelos_avanzados.ipynb`
**Matriz:** `data/processed/matriz_frecuencia_v2.csv` · **Offset:** `log_exposicion_v2` · **Semilla:** 42
**Partición:** train 2022–2024 / test 2025–2026, sin alterar

---

## 1. El encargo y la respuesta en una frase

NB05 corrigió la exposición y, al hacerlo, mejoró la calibración del Random Forest
de 0.8196 a 1.0057 pero le costó capacidad de ordenar distritos: la correlación de
Spearman en el tercio de mayor siniestralidad cayó de 0.9732 a 0.9456. NB07
recupera esa capacidad —y la supera— **sacando la exposición del conjunto de
variables predictoras y colocándola como offset en XGBoost**.

---

## 2. Qué decidí y por qué

### 2.1 Añadí una partición interna de validación dentro de train

**Qué hice.** Ajuste sobre 2022–2023, validación sobre 2024. Toda la selección de
hiperparámetros y de especificación se resolvió ahí. El conjunto de prueba se usó
una sola vez, al final.

**Por qué.** Optuna prueba decenas de configuraciones y se queda con la mejor. Si
esa elección se hiciera mirando el test, el número resultante sería el máximo de
ochenta mediciones ruidosas sobre el mismo conjunto, y la comparación contra
Etapa 1 y NB05 —cuyos hiperparámetros se fijaron sin mirarlo— quedaría sesgada a
favor de NB07.

**Lo que no cambié.** La frontera train/test de NB03 está intacta. La partición
interna vive dentro de train y no toca esa frontera.

### 2.2 La función objetivo de Optuna no es el MAE, ni el RMSE a secas

**Qué hice.** `objetivo = RMSE × (1 + |ln(calibración)|)`.

**Por qué el MAE queda fuera.** Con 89.7% de celdas en cero, el predictor que
minimiza el error absoluto es el que predice casi siempre cerca de cero. Un
modelo baja su MAE simplemente prediciendo menos. Es el criterio que
reintroduciría el Problema A que NB05 corrigió.

**Por qué tampoco el RMSE solo.** El RMSE es menos sensible que el MAE al exceso
de ceros, pero no es neutral: su mínimo se sitúa algo por debajo de la
calibración exacta. El notebook lo demuestra con los propios ensayos: en la
variante de offset, el RMSE puro habría elegido una configuración con 0.28% menos
de error y una calibración de 0.9851 en lugar de 0.9998.

**Por qué esa forma concreta.** El logaritmo hace que predecir el doble y
predecir la mitad se penalicen igual. La penalización es multiplicativa, de modo
que no hay que recalibrar el coeficiente cuando cambia la escala del RMSE. Con
λ = 1, una desviación de calibración del 5% añade 4.9% al objetivo, mientras las
diferencias de RMSE entre configuraciones razonables son del orden del 1%: la
calibración domina, que es lo que la jerarquía de NB05 prescribe.

### 2.3 Probé dos formas de meter la exposición en XGBoost, y ahí está el hallazgo

- **Variante A, exposición como variable predictora.** Nueve columnas, igual que
  el Random Forest de NB04 y NB05. El algoritmo decide dónde partirla.
- **Variante B, exposición como offset.** Ocho columnas; la exposición entra por
  `base_margin`, que XGBoost suma al predictor lineal antes del primer árbol. Con
  objetivo `count:poisson` el enlace es logarítmico, así que reproduce
  exactamente la función que el offset cumple en un GLM.

**Por qué esperaba que importara.** El diagnóstico de la pérdida ordinal de NB05
es que la corrección de exposición atraviesa la estructura de particiones: un
desplazamiento uniforme de `log_exposicion` hace que cada celda cruce un umbral
distinto y aterrice en una hoja cuyo valor no guarda relación proporcional con el
anterior. La corrección de escala se convierte en perturbación del orden. Si eso
es cierto, sacar la exposición de las particiones debería resolverlo.

**Resultado.** Las dos variantes comparten algoritmo, presupuesto, función
objetivo y partición; lo único que las separa es dónde entra la exposición. La de
offset alcanza 0.9775 de Spearman en el tercil superior; la de variable, 0.9520.
La diferencia es del mismo orden que la pérdida que NB05 había sufrido y aparece
exactamente donde el diagnóstico la predecía.

### 2.4 Conservé `poblacion_baja`

NB05 la señaló como candidata a eliminación por su importancia Gini de 0.0001.
Medí con y sin ella: la diferencia en RMSE es de −0.026% en una variante y
+0.046% en la otra, y en la de offset el objetivo empeora un 0.31% al quitarla.
La atribución SHAP le asigna el 2.6% del peso, por encima de `es_feriado`.

La lección es sobre la métrica, no sobre la variable: la importancia Gini mide
con qué frecuencia el algoritmo partió por una variable, no cuánta información
aporta. Se conserva, y el esquema de ocho variables sigue siendo el mismo de
NB04 y NB05.

---

## 3. Tabla comparativa final

Todos los modelos evaluados sobre el mismo conjunto de prueba (2025-01 a
2026-06), con el mismo código y en la misma ejecución. Las líneas base se
recalcularon cargando los `.pkl` de cada etapa, y reproducen exactamente las
cifras publicadas en `nb04_metricas_baseline.json` y `nb05_metricas.json`.

| Etapa | Modelo | MAE | RMSE | Calibración | Spearman global | Spearman tercil alto | AIC |
|---|---|---|---|---|---|---|---|
| Etapa 1 | Binomial Negativa | 0.201136 | 0.435272 | 0.7954 | 0.8857 | 0.8181 | 289,004.73 |
| Etapa 1 | Random Forest | 0.191815 | 0.424042 | 0.8196 | 0.9903 | 0.9732 | — |
| NB05 | Poisson | 0.212615 | 0.429055 | 0.9647 | 0.8864 | 0.8136 | 289,438.77 |
| NB05 | Binomial Negativa | 0.211235 | 0.429701 | 0.9422 | 0.8861 | 0.8181 | 288,836.57 |
| NB05 | Random Forest | 0.205268 | 0.418678 | 1.0057 | 0.9373 | 0.9456 | — |
| NB07 | ZIP | 0.212194 | 0.429449 | 0.9539 | 0.8864 | 0.8220 | 287,829.95 |
| NB07 | ZINB | 0.212202 | 0.429652 | 0.9511 | 0.8871 | 0.8187 | **287,585.28** |
| NB07 | Random Forest regularizado | 0.203808 | 0.416231 | 0.9939 | 0.9372 | 0.9421 | — |
| NB07 | XGBoost (exposición variable) | 0.203085 | 0.414302 | 0.9537 | 0.9473 | 0.9520 | — |
| **NB07** | **XGBoost (exposición offset)** | **0.200749** | **0.412055** | **0.9670** | **0.9708** | **0.9775** | — |

**Estabilidad temporal** (deriva de la calibración a lo largo de los 18 meses del
horizonte, según ajuste lineal sobre las razones mensuales):

| Modelo | σ mensual | Deriva 18 meses |
|---|---|---|
| Random Forest Etapa 1 | 0.0708 | −0.1075 |
| Random Forest NB05 | 0.0395 | −0.0408 |
| Random Forest regularizado NB07 | 0.0411 | −0.0451 |
| XGBoost exposición variable | 0.0390 | −0.0629 |
| **XGBoost exposición offset** | **0.0378** | **−0.0008** |

---

## 4. ¿Se resolvió el compromiso calibración / orden?

**Sí.**

| | Calibración | Spearman tercil alto | RMSE |
|---|---|---|---|
| Random Forest Etapa 1 | 0.8196 | 0.9732 | 0.424042 |
| Random Forest NB05 | 1.0057 | 0.9456 | 0.418678 |
| **XGBoost offset NB07** | **0.9670** | **0.9775** | **0.412055** |

- **Ordenamiento:** 0.9775, que supera tanto el 0.9456 de NB05 (+0.0319) como el
  0.9732 del Random Forest de Etapa 1 (+0.0043). Ese 0.9732 lo alcanzaba un
  modelo que subestimaba el nivel en un 18%.
- **Calibración:** 0.9670, dentro de la banda de ±5% que el notebook declara
  admisible en su Sección 3.5. La banda no es convencional: NB05 documentó en su
  Sección 6.5 que el factor de crecimiento, anclado en enero de 2022, introduce
  una desviación esperada del 2 al 3% que ningún modelo entrenado sobre esa
  exposición puede eliminar. Exigir menos que el sesgo conocido de la propia
  especificación sería seleccionar por ruido.
- **RMSE:** 0.412055, el menor de las tres etapas, un 1.58% por debajo de NB05.

**Resultado adicional que no buscaba y es el más limpio.** La deriva de
calibración pasa de −0.0408 a −0.0008: dos órdenes de magnitud menos. NB05 había
declarado como limitación que el Random Forest agota el rango de
`log_exposicion` que conoce y por eso exige reentrenamiento periódico. Con la
exposición como offset la limitación desaparece por construcción, no por ajuste:
el factor de crecimiento es un sumando exacto del predictor lineal y se traslada
íntegro a la predicción por lejos que llegue el horizonte.

---

## 5. Qué salió peor de lo esperado

### 5.1 La regularización del Random Forest no funcionó — hipótesis refutada

Era una de las dos vías que el encargo planteaba. La idea: si la pérdida ordinal
viene de la finura de las particiones sobre `log_exposicion`, hojas más grandes y
menos profundidad deberían atenuarla.

La rejilla de ocho configuraciones mejora el RMSE hasta 0.416231 y la calibración
hasta 0.9939 —ambas mejores que NB05— pero la correlación en el tercil superior
queda en **0.9421, por debajo del 0.9456 de NB05**. El problema no era la finura
de las particiones sino el hecho mismo de particionar por la exposición.

La rejilla dejó además un patrón instructivo: al reducir el número de hojas el
RMSE mejora y la calibración se deteriora de forma monótona, de 1.0103 a 0.9727.
Regularizar hace exactamente lo que el MAE premiaba en NB05 — bajar las
predicciones.

### 5.2 `min_child_weight` se fue al extremo bajo

Di a Optuna un rango de 1 a 200 esperando que, si la regularización era la vía,
se concentrara arriba. Eligió 4.2 y 3.1 en las dos variantes. Exigir hojas
pobladas no mejora el objetivo. Es la misma refutación que la anterior, desde el
otro algoritmo.

### 5.3 ZIP y ZINB: mucho ajuste, ninguna predicción

Producen la mayor mejora de verosimilitud del proyecto: el AIC baja de
288,836.57 a **287,585.28**, 1,251 puntos, un orden de magnitud más de lo que
NB05 había ganado corrigiendo la exposición. El contraste de Vuong es concluyente
en las tres comparaciones (V = 19.44, 19.20 y 6.75).

Y no sirve para predecir: el RMSE sobre prueba coincide con el de los modelos no
inflados en el cuarto decimal, y la capacidad ordinal se queda en 0.82, el nivel
de todos los modelos lineales del proyecto.

La explicación es que el AIC mide el ajuste de la distribución completa y el RMSE
el error de la media condicional. Un modelo inflado describe mucho mejor la
forma —cuántos ceros, cómo se reparten los positivos— sin desplazar la media.
Sirve para intervalos de predicción, no para ordenar distritos.

**Cautela que hay que declarar si preguntan.** El modelo estima que el 42.6% de
las celdas pertenece al componente inflado. La lectura literal —que en el 43% de
los casos el siniestro no podía ocurrir— no es creíble. Lo que ese componente
absorbe es la heterogeneidad territorial que NB05 documentó como Problema B. La
etiqueta de «cero estructural» es una conveniencia del modelo, no un hallazgo.

### 5.4 El tercil medio sigue roto

| Modelo | Tercil alto | Tercil medio | Tercil bajo |
|---|---|---|---|
| Random Forest Etapa 1 | 0.9732 | 0.8794 | 0.8888 |
| Random Forest NB05 | 0.9456 | 0.6832 | 0.7388 |
| XGBoost offset NB07 | 0.9775 | 0.6470 | 0.8003 |

Todos los modelos entrenados sobre la exposición corregida ordenan mal el tercio
intermedio. NB07 recupera la cabeza del ranking y algo de la cola, y no toca el
medio. Es el problema abierto que se traslada a NB08.

### 5.5 La calibración es peor que la de NB05

0.9670 frente a 1.0057. Está dentro de la banda y muy por encima del 0.8196 de la
Etapa 1, pero si el uso fuera estimar volúmenes absolutos —dimensionar un
presupuesto, por ejemplo— el Random Forest regularizado, con 0.9939, sería mejor
elección. Queda documentado como alternativa.

### 5.6 Inferencia limitada en los modelos inflados

La inversión del hessiano falla en el ZINB y en el ZIP de inflación constante, de
modo que esos ajustes no producen errores estándar y sus coeficientes no admiten
contraste de significación. Las predicciones son utilizables. La Binomial
Negativa de NB05 conserva la función inferencial del proyecto.

---

## 6. Archivos

### 6.1 Creados

| Archivo | Tamaño | Contenido |
|---|---|---|
| `notebooks/NB07_optimizacion_modelos_avanzados.ipynb` | ~0.9 MB | El notebook ejecutado |
| `models/frecuencia_xgboost_offset_v2.pkl` | 0.06 MB | **Modelo seleccionado.** XGBoost, exposición como offset |
| `models/frecuencia_xgboost_v2.pkl` | 0.48 MB | XGBoost, exposición como variable predictora |
| `models/frecuencia_random_forest_nb07.pkl` | 62.40 MB | Random Forest regularizado (`min_samples_leaf=20`) |
| `models/frecuencia_zip_v2.pkl` | 2.06 MB | ZIP, inflación territorial |
| `models/frecuencia_zinb_v2.pkl` | 2.06 MB | ZINB, inflación territorial, α estimado = 0.3584 |
| `reports/results/nb07_metricas.json` | 0.01 MB | Métricas de los diez modelos comparados |
| `reports/results/nb07_especificacion.json` | 0.01 MB | Orden de columnas y parámetros de carga de cada modelo nuevo |
| `reports/results/nb07_optuna.db` | 0.23 MB | Estudios de Optuna (SQLite, reanudables) |
| `reports/figures/nb07_01_optuna_trayectoria.png` | | Trayectoria de la búsqueda y descomposición RMSE/calibración |
| `reports/figures/nb07_02_rf_regularizacion.png` | | Complejidad del bosque frente a error y calibración |
| `reports/figures/nb07_03_calibracion_mensual.png` | | Calibración mes a mes y deriva |
| `reports/figures/nb07_04_compromiso_calibracion_orden.png` | | El compromiso de NB05 y dónde cae cada modelo |
| `reports/figures/nb07_05_shap_resumen.png` | | Importancia y distribución de atribuciones |
| `reports/figures/nb07_06_shap_dependencia.png` | | Dependencia de las variables continuas |
| `NB07_RESUMEN.md` | | Este documento |

**Modificado:** `requirements.txt`, con un comentario que documenta que XGBoost
necesita OpenMP en el sistema (`brew install libomp` en macOS). Sin él,
`import xgboost` falla al cargar `libxgboost.dylib`. No se añadió ninguna
dependencia: `xgboost==2.1.1`, `optuna==4.0.0` y `shap==0.46.0` ya estaban
fijados.

### 6.2 Versionado

**Ningún artefacto de NB07 supera los 100 MB**, de modo que todos se versionan y
no hizo falta tocar `.gitignore`. El mayor es el Random Forest regularizado, con
62.40 MB — menos de la mitad que el de NB05 (122.67 MB, no versionado) porque
`min_samples_leaf=20` reduce el número de hojas un 57%.

### 6.3 No se tocó nada previo

El notebook verifica en su Sección 10.4 que ninguno de los quince artefactos de
NB04 y NB05 —modelos, matrices y archivos de métricas— haya sido modificado
durante la ejecución, y falla si detecta lo contrario. La verificación pasó.

Tampoco se modificó `src/config.py`. Las funciones que NB07 necesitaba y que no
existían allí se definieron dentro del notebook, porque corresponden a decisiones
analíticas de esta etapa y no a plomería reutilizable.

### 6.4 Cómo cargar el modelo seleccionado

Atención a un detalle que el `.pkl` no contiene:

```python
import joblib, xgboost as xgb, numpy as np, json

espec = json.load(open("reports/results/nb07_especificacion.json"))
info = espec["modelos"]["frecuencia_xgboost_offset_v2.pkl"]

booster = joblib.load("models/frecuencia_xgboost_offset_v2.pkl")
X = df[info["columnas"]]                       # 8 columnas, orden normativo
margen = df["log_exposicion_v2"] + info["log_tasa_base"]
pred = booster.predict(xgb.DMatrix(X, base_margin=margen))
```

`log_tasa_base` (−12.936335) **no forma parte del objeto serializado**. Sin él
las predicciones se desvían varios órdenes de magnitud sin producir ningún error.
Está registrado en `nb07_especificacion.json`, junto con el orden de columnas de
los cinco modelos nuevos.

---

## 7. Pendiente para NB08

1. **El tercil medio.** Spearman de 0.6470 frente al 0.8794 de Etapa 1. Es el
   problema abierto. Conviene determinar si el deterioro proviene de la
   corrección de exposición de NB05 —afecta por igual a todos los modelos que la
   usan— o de la ausencia de variables que diferencien distritos de
   siniestralidad intermedia.

2. **La heterogeneidad territorial (Problema B de NB05).** Sigue sin corregir.
   NB05 estableció que ninguna variable disponible la explica y que hacen falta
   fuentes externas: parque vehicular por distrito, densidad de red vial, aforos
   del VMT. `osmnx==1.9.4` ya está en `requirements.txt`; la densidad de red vial
   es la candidata con menor coste de obtención.

3. **Aprovechar ZINB para lo que sirve.** Describe la distribución completa mucho
   mejor que cualquier modelo previo del proyecto. Un prototipo que comunique
   riesgo con intervalos de predicción, y no solo con un conteo esperado, tiene
   ahí una base que NB07 no explotó.

4. **Ampliar la búsqueda de la variante de offset.** Es la única de las dos que
   no dio señales de haber convergido: su mejor valor aparece en el ensayo 38 de
   40, y en el ensayo 20 todavía estaba a un 0.69% del final. La variante de
   exposición como variable sí convergió —en el ensayo 20 estaba a un 0.042%—.
   El estudio queda persistido en `reports/results/nb07_optuna.db`: elevar
   `N_ENSAYOS` y reejecutar la celda conserva los ochenta ensayos existentes y
   añade los nuevos. El margen que quede está acotado por la dispersión total
   entre ensayos, del 1.8%.

5. **Modelo de referencia para el prototipo.** XGBoost con exposición como
   offset, por ordenamiento, RMSE y estabilidad temporal. El Random Forest
   regularizado queda como alternativa cuando importe la magnitud absoluta. La
   Binomial Negativa de NB05 conserva la función inferencial.

---

## 8. Reproducción

```bash
# Dependencia de sistema para XGBoost en macOS
brew install libomp

# Regenerar matriz y modelos de NB05 si faltan (no versionados por tamaño)
jupyter nbconvert --to notebook --execute --inplace notebooks/NB05_exposicion_y_tendencia.ipynb

# Ejecutar NB07
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/NB07_optimizacion_modelos_avanzados.ipynb
```

Todo el notebook usa `cfg.SEED = 42`. La celda de persistencia se niega a
sobrescribir un artefacto existente, de modo que una reejecución requiere borrar
antes los cinco `.pkl` de NB07. Los estudios de Optuna, en cambio, se reanudan:
si el `.db` ya contiene los cuarenta ensayos de cada variante, la celda no repite
ninguno y los resultados son idénticos.
