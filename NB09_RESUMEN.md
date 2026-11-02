# NB09 — Resumen para la defensa

**Notebook:** `notebooks/NB09_calibracion_y_submodelo_severidad.ipynb`
**Rama:** `feat/nb09` · **Semilla:** 42 · **Matriz:** `matriz_severidad.csv` (89,946 siniestros)
**Split:** train 2022–2024 / test 2025–2026, fijado en NB03 y no alterado

---

## 1. Qué responde este notebook

NB04 dejó la severidad con dos clasificadores, un F1-macro de 0.3247 y una limitación
escrita. NB09 retoma esa tarea después de que las Etapas 5 a 8 se ocuparan de la
frecuencia, y responde dos preguntas.

**¿Son honestas las probabilidades de NB04?** No, y por una razón que estaba a la vista:
ambos modelos usan `class_weight='balanced'`, que les pide razonar como si las tres clases
fueran igual de frecuentes.

**¿Cuánto aportan las variables de víctima cuando su uso es legítimo?** Mucho. Sin ellas,
el submodelo condicional está prácticamente en el azar.

---

## 2. Sección 2 — Calibración

### 2.1 El hallazgo principal

La probabilidad media que emite la Regresión Logística es 0.3412 / 0.3345 / 0.3243 para
las tres clases. Las frecuencias reales son 0.5158 / 0.4281 / 0.0562.

| Modelo | Distancia a la frecuencia real | Distancia a la uniforme (⅓) |
|---|---|---|
| Logística | 0.1787 | **0.0060** |
| Random Forest | 0.1157 | 0.0691 |

La logística no está *desviada* respecto de la realidad: está **alineada con la uniforme**.
Eso no es un defecto, es `class_weight='balanced'` haciéndose visible. Por eso el notebook
no reporta «ECE 0.1787» a secas: sería describir el diseño de NB04 como si fuera un error.

### 2.2 Las dos expectativas de la literatura fallan

Me pediste verificar en vez de asumir. Las dos se cayeron:

| Expectativa | Predicción | Observado |
|---|---|---|
| El bosque comprime hacia el centro | rango del RF menor | **RF 0.5595** contra **logística 0.1445** |
| La logística está bien calibrada de fábrica | sesgo pequeño | Sesgo **+0.2645** en la zona media, **+0.4000** en la alta |

La que comprime es la logística, por un factor de casi cuatro. Sobre 32 variables casi
todas binarias y con pesos balanceados, su predictor lineal tiene recorrido corto y todo
se apiña en torno a un tercio: nunca asigna a `fatal` menos de 0.2097.

El Random Forest, con hojas de una observación, produce probabilidades extremas. Y su
curva para `fatal` es **monótona creciente** (0.0545 → 0.1293) mientras la de la logística
oscila sin dirección: **ordena mejor aunque su nivel absoluto esté igual de mal**.

### 2.3 La corrección: aceptada en ambos

| Modelo | ECE gemelo | ECE + Platt | Reducción | Brier antes | Brier después |
|---|---|---|---|---|---|
| Logística | 0.1748 | **0.0087** | 95.0% | 0.2204 | 0.1818 |
| Random Forest | 0.1706 | **0.0030** | 98.3% | 0.2303 | 0.1830 |

El criterio D3 exigía 20% de reducción sin empeorar el Brier. Se obtuvo 95–98%, y el
Brier mejoró también.

**Por qué esa magnitud es el resultado y no solo un número bueno:** una transformación de
**dos parámetros** elimina casi todo el error. Eso significa que el desajuste no tenía
estructura —era un desplazamiento en escala logit, que es exactamente la forma de un
cambio de prior—. Si el problema hubiera sido de especificación del modelo, una sigmoide
de dos parámetros no habría podido corregirlo.

Elegí Platt y no isotónica por eso mismo: la familia de funciones corresponde al mecanismo
que se quiere corregir. La isotónica es más flexible pero tendría que aprender esa forma
desde cero, con ~1,200 fatales en el bloque de calibración.

---

## 3. Sección 3 — El submodelo condicional

### 3.1 El argumento anti-fuga, para decirlo en la defensa

> "No relajamos la regla. Cambiamos la pregunta hasta que la regla dejó de tener objeto.
>
> La fuga consistía en que `grupo_vulnerable` valía `Sin victimas` en exactamente los
> 46,841 casos de `solo_danos`: era una codificación de la clase, no un predictor. Al
> condicionar sobre que hubo víctima, esa categoría no existe —por construcción, no por
> filtrado— y la variable pasa a informar sobre **quién** fue la víctima, no sobre si la
> hubo."

**Y lo verifiqué, porque el argumento lo exige.** Si `grupo_vulnerable` se hubiera
definido mirando quién murió, la circularidad volvería por la puerta de atrás. Fui a NB03:

- `grupo_vulnerable` = el **más vulnerable presente**, por prioridad fija peatón > ciclista > motociclista > otro.
- `rango_etario` = el **del conductor**; si no hay, el de la primera víctima.

Ninguna consulta el desenlace. Ambas son observables sin saber qué pasó. El argumento se
sostiene.

### 3.2 El número que justifica el notebook

Validación 2024:

| Esquema | AUC-PR | Lift sobre el piso | ROC-AUC |
|---|---|---|---|
| Sin variables de víctima (9) | 0.1413 | 1.080 | 0.5296 |
| Con variables de víctima (11) | **0.2388** | **1.825** | **0.6853** |
| *Piso (prevalencia)* | *0.1308* | *1.000* | *0.500* |

**Δ AUC-PR = +0.0975, IC 95% [+0.0815, +0.1151]** sobre 2,000 remuestreos pareados. Las
2,000 réplicas dieron diferencia positiva.

Confirmación en el test, usado una sola vez:

| Esquema | AUC-PR | Lift | ROC-AUC | Precisión | Recall | F1 fatal |
|---|---|---|---|---|---|---|
| Sin víctima | 0.1348 | 1.162 | 0.5357 | 0.1309 | 0.4365 | 0.2014 |
| Con víctima | **0.2163** | **1.865** | **0.6835** | **0.1818** | **0.6247** | **0.2816** |
| *Piso* | *0.1160* | *1.000* | *0.500* | | | |

**Δ en test: +0.0815, IC 95% [+0.0689, +0.0942].** Excluye el cero.

En términos operativos: detecta **1,185 de 1,897 fatalidades** contra 828 del esquema
reducido, y con **menos** falsos positivos (5,333 contra 5,496). Gana en las dos
dimensiones, que no es lo habitual al mover un clasificador.

### 3.3 El resultado que no esperaba

**Sin las variables de víctima el submodelo está casi en el azar.** AUC-PR 0.1348 sobre un
piso de 0.1160; ROC-AUC 0.5357, a tres centésimas de la diagonal.

Yo esperaba que `tipo_siniestro` y `tipo_vehiculo` aportaran más: un volcamiento no es una
colisión leve, una moto no protege como un auto. Aportan, pero muy poco **una vez que se
condiciona sobre que hubo víctima**. La severidad, dado que alguien resultó afectado,
depende sobre todo de quién fue ese alguien.

El control lo confirma: el esquema de 9 variables contra una puntuación constante da
Δ +0.0105, IC [+0.0031, +0.0190]. Supera al azar técnicamente, pero por un orden de
magnitud menos que el contraste principal.

### 3.4 Métricas: qué elegí y por qué

**Principal: AUC-PR de `fatal`.** Libre de umbral (la comparación central es entre dos
especificaciones; si dependiera de un umbral habría que elegirlo, y esa elección
contaminaría lo que se quiere medir), centrada en la minoría, y con piso interpretable: la
prevalencia.

Descarté **ROC-AUC** como principal: con 12% de positivos es optimista. Descarté **F1 de
fatal**: depende del umbral. Ambas se reportan como secundarias.

**El número decisivo** no es ninguna por separado, sino el Δ con intervalo bootstrap
pareado. Sin intervalo, una mejora puntual no es evidencia.

---

## 4. Sección 4 — Interpretabilidad

### 4.1 SHAP confirma la Sección 3 por otra vía

| Variable | \|SHAP\| medio | Peso | Impureza |
|---|---|---|---|
| `grupo_vulnerable` | 0.06359 | **46.3%** | 0.2305 |
| `tipo_vehiculo` | 0.01934 | 14.1% | 0.0817 |
| `rango_etario` | 0.01757 | 12.8% | 0.2186 |

Las dos variables de víctima reúnen el **59.1%** de la atribución.

Detalle instructivo: por impureza, `grupo_vulnerable` y `rango_etario` están casi empatadas
(0.2305 y 0.2186). Por SHAP, la primera pesa más del triple. La impureza premia a las
variables con más categorías, y `rango_etario` tiene trece contra cuatro. Es el mismo
fenómeno que NB07 encontró con `poblacion_baja`, pero en dirección contraria.

### 4.2 Una advertencia que hay que tener lista

**Me pasaste que NB02 halló motociclistas 42% de los fallecidos, peatones 34.2%, letalidad
del peatón 18%. Verifiqué contra NB02 y es correcto.** Pero sobre `matriz_severidad` el
orden **se invierte**:

| Grupo | % de fatales por siniestro | Letalidad por siniestro |
|---|---|---|
| Peatón | 42.47% | 0.2294 |
| Motociclista | 36.94% | 0.1500 |

No es una contradicción: **es la unidad de análisis**. NB02 cuenta víctimas
(`victimas.csv`); la matriz cuenta siniestros, y `grupo_vulnerable` es «el más vulnerable
presente». Un siniestro con peatón y motociclista se etiqueta Peatón; un siniestro de moto
aporta a menudo dos víctimas.

**Lo comparable es el ordenamiento, no los porcentajes.** Si alguien compara 42.0% contra
42.47% y dice «se reproduce», está comparando cosas distintas.

Y el ordenamiento sí se reproduce:

- `grupo_vulnerable`: concordancia **exacta** entre el orden de las atribuciones SHAP y el
  de la letalidad observada, en los cuatro grupos.
- `rango_etario`: **Spearman 0.9560** entre atribución media y letalidad, sobre trece
  categorías. Máximo en 71+ (letalidad 0.2982), mínimo en 0–15 (0.0900).

Esto no descubre nada nuevo: NB02 ya lo sabía. Lo que establece es que **el submodelo
aprendió la estructura correcta y no un atajo**.

### 4.3 El caso LIME

Falso negativo más extremo: `SV-2025-076926`, Cuscatancingo, 7-12-2025, tarde. Desenlace
fatal, probabilidad asignada **0.0053** contra un umbral de 0.1494.

**Todas las variables empujaron hacia «no fatal»**: `grupo_vulnerable = Otro` (−0.1319),
`rango_etario = 16-20` (−0.0269), `tipo_vehiculo = Automovil` (−0.0220), colisión, vía
seca, sin lluvia. Es el perfil de menor letalidad del conjunto (0.0407).

El modelo no falló por falta de información sobre este caso: falló porque el caso es
atípico dentro de su categoría. Son **712 de 1,897** fatalidades en esa situación. Para
distinguirlas harían falta velocidad de impacto, uso de cinturón o tiempo de respuesta
médica, que el conjunto no contiene.

R² del ajuste local: 0.4901. Ni el entorno inmediato del caso se aproxima bien con un
modelo lineal.

---

## 5. Sección 5 — Para qué sirve

**Coincido con tu lectura y la sostengo con los números.** El submodelo no entra en el
prototipo predictivo, y no por una limitación corregible: `grupo_vulnerable` y
`rango_etario` se conocen tras identificar a las víctimas. Pedirle una predicción ex ante
es pedirle que anticipe su propio condicionante.

Tres usos que el desempeño sostiene:

**1. Orientar política pública.** Es el mejor respaldado y no exige acertar ningún caso
individual. Un peatón involucrado en un siniestro tiene **5.6 veces** la probabilidad de
morir que un ocupante de vehículo cerrado (0.2294 contra 0.0407). El aporte del submodelo
es confirmar que esa asociación **persiste controlando por el resto de variables**: no es
un artefacto de que los peatones aparezcan en siniestros de otro tipo.

**2. Estratificar el análisis retrospectivo.** Los 712 falsos negativos son exactamente el
conjunto que merece revisión cualitativa: son las muertes que las variables disponibles no
explican. El filtro sirve aunque la precisión sea baja, porque el criterio es «señalar lo
anómalo», no «acertar».

**3. Con reservas: acompañar el mapa de frecuencia** con la composición histórica de
víctimas del distrito. Pero eso **no es usar el submodelo**: es usar la tabla de letalidad
que el submodelo ayudó a validar. Presentarlo como predicción de gravedad sería el error
que 5.1 descarta.

**Un uso a descartar explícitamente:** con precisión 0.1818, no sirve para priorizar
respuesta de emergencia caso a caso. De cada seis siniestros que señala, cinco no son
fatales.

---

## 6. Decisiones discutibles, declaradas

**D7 — el submodelo sin `class_weight`.** Es la más discutible y la declaro como tal.
Mantenerlo habría dado continuidad con NB04, pero haría que las probabilidades del
submodelo fueran deliberadamente incorrectas, lo que contradice el propósito del notebook.
Prioricé la coherencia interna y dejé constancia del costo: **las métricas del submodelo no
son comparables en igualdad de condiciones con las de los clasificadores de tres clases.**

**Los gemelos de la Sección 2.** Los modelos de NB04 se entrenaron sobre 2022–2024
completo, así que dentro de train no hay ningún bloque externo para ellos. Reinstancié los
clasificadores sobre 2022, ajusté el calibrador sobre 2023 y decidí sobre 2024. Queda
demostrado que **el procedimiento funciona**, no que los artefactos de NB04 estén
calibrados.

**Tres bloques en la Sección 2, dos en la Sección 3.** No es inconsistencia: la Sección 2
ajusta un calibrador y necesita un bloque más que la Sección 3, que solo selecciona entre
dos especificaciones.

---

## 7. Limitaciones

| Limitación | Evidencia | Implicación |
|---|---|---|
| La corrección se demostró sobre gemelos | Entrenados sobre 2022, un tercio de train | El procedimiento está verificado; los modelos publicados no están calibrados |
| El submodelo sobreestima el nivel | Media predicha 0.1391 contra prevalencia 0.1160 | La letalidad cayó de 0.1526 (2022) a 0.1140 (2025). Es desplazamiento temporal, del tipo que NB05 corrigió para frecuencia |
| Precisión baja en absoluto | 0.1818 con el umbral de operación | Inutilizable caso a caso |
| Variables ex post | Se conocen tras identificar víctimas | No admite uso predictivo. No es corregible |
| Rompe continuidad con NB04 | Decisión D7 | Métricas no comparables en igualdad de condiciones |
| Unidad distinta de NB02 | 0.2294 por siniestro contra 0.180 por víctima | Los porcentajes no son intercambiables; el ordenamiento sí |
| 712 fatalidades sin explicar | Falsos negativos en perfiles de bajo riesgo | Faltan velocidad de impacto, cinturón, tiempo de respuesta médica |
| Conjunto sintético | Calibrado en NB01 sobre agregados ONASEVI/FONAT | La validación externa exige registros reales |

---

## 8. Archivos

### Creados

| Archivo | Tamaño | Contenido |
|---|---|---|
| `notebooks/NB09_calibracion_y_submodelo_severidad.ipynb` | ~0.5 MB | 43 celdas, 19 de código, 3 figuras |
| `models/severidad_condicional_nb09.pkl` | 15.43 MB | **Submodelo principal** (11 variables) |
| `models/severidad_condicional_sin_victima_nb09.pkl` | 12.33 MB | Control de ablación (9 variables) |
| `models/severidad_calibrador_logistica_multinomial_nb09.pkl` | 0.00 MB | Gemelo logístico calibrado |
| `models/severidad_calibrador_random_forest_nb09.pkl` | 28.74 MB | Gemelo RF calibrado |
| `reports/results/nb09_metricas.json` | | Todas las métricas |
| `reports/results/nb09_especificacion.json` | | Esquema normativo de columnas |
| `reports/results/nb09_decisiones.json` | | D1–D8 con marca de tiempo |
| `reports/results/nb09_hashes_iniciales.json` | | Referencia de integridad |
| `reports/figures/nb09_01_fiabilidad_3clases.png` | | Diagramas de fiabilidad y ocupación de bins |
| `reports/figures/nb09_02_submodelo_calibracion_pr.png` | | Fiabilidad del submodelo y curva precisión-recall |
| `reports/figures/nb09_03_shap_submodelo.png` | | Importancia y distribución de atribuciones |
| `NB09_RESUMEN.md` | | Este documento |

Ningún artefacto supera los 100 MB. No hizo falta tocar `.gitignore`.

### Nada previo fue modificado

La celda 1.1 registra el SHA-256 de **19 artefactos** de NB01 a NB08 y la 6.2 los vuelve a
verificar. El cierre imprime:

```
Integridad de artefactos previos: verificada (19 archivos intactos)
```

No se modificó `src/config.py` ni ningún notebook previo. No se agregó ninguna dependencia.

### Cómo cargar el submodelo

```python
import json, joblib, pandas as pd

espec = json.load(open("reports/results/nb09_especificacion.json"))
info = espec["modelos"]["severidad_condicional_nb09.pkl"]

m = joblib.load("models/severidad_condicional_nb09.pkl")
X = df[info["columnas"]].copy()
for c in info["categoricas"]:                       # cat.codes, mapeo del ajuste
    cats = info["categorias_por_columna"][c]
    X[c] = pd.Categorical(X[c].astype(str), categories=cats).codes
p = m.predict_proba(X)[:, list(m.classes_).index(1)]
decision = p >= info["umbral_operacion"]            # 0.1494, NO está dentro del .pkl
```

Dos trampas registradas en el JSON: el **umbral** no viaja dentro del objeto, y las
`classes_` de los modelos de NB04 están en orden **alfabético**, que no coincide con el
orden declarado en aquel notebook. Indexar `predict_proba` por posición sin mirar
`classes_` sería un error silencioso.

---

## 9. Qué queda pendiente

1. **Calibrar los artefactos de NB04 de verdad**, con validación cruzada sobre todo el
   entrenamiento en lugar de gemelos sobre un año. El procedimiento ya está verificado.
2. **Corregir el desplazamiento temporal de la letalidad** en el submodelo, con la misma
   lógica que NB05 aplicó a la frecuencia.
3. **Revisar cualitativamente los 712 falsos negativos.** Es el conjunto donde las
   variables disponibles fallan, y por tanto el que indica qué habría que medir.
4. **Decidir si la tabla de letalidad por grupo entra en el prototipo** como dato
   descriptivo del distrito, nunca como predicción.
