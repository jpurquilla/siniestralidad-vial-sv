# Resumen de NB09 y NB10 para el informe final

**Proyecto:** Análisis de patrones espacio-temporales de accidentes de tránsito en El Salvador
**Etapa:** 2 — Modelos avanzados · **Metodología:** CRISP-DM
**Notebooks cubiertos:** NB09 (severidad) y NB10 (clustering espacio-temporal)

Documento de síntesis. Los detalles completos están en `NB09_RESUMEN.md`, `NB10_RESUMEN.md`
y en los propios notebooks.

---

## 1. Qué aporta cada notebook en una línea

| Notebook | Aporte |
|---|---|
| **NB09** | Las probabilidades de los clasificadores de severidad de NB04 no son honestas, pero su desajuste es corregible con dos parámetros; y condicionando a que hubo víctima, las variables de víctima resultan legítimas y aportan de forma medible |
| **NB10** | Se cumple el compromiso del anteproyecto con ST-DBSCAN y se establece, con evidencia, que la capa espacial del conjunto carece de la resolución necesaria para identificar puntos negros |

Los dos cierran temas abiertos: NB09 la limitación que NB04 había declarado por escrito;
NB10 el compromiso de marco teórico del anteproyecto.

---

## 2. NB09 — Calibración de severidad y submodelo condicional

### 2.1 Las probabilidades de NB04 están alineadas con la uniforme, no con la realidad

Ambos clasificadores de NB04 usan `class_weight='balanced'`, que les pide optimizar como si
las tres clases fueran equiprobables. La consecuencia es medible:

| Modelo | Distancia a la frecuencia real | Distancia a la uniforme (⅓) |
|---|---|---|
| Regresión Logística | 0.1787 | **0.0060** |
| Random Forest | 0.1157 | 0.0691 |

La logística no está desviada respecto de la realidad: está **alineada con la uniforme**.
No es un defecto de implementación sino el diseño de NB04 haciéndose visible, y el informe
debe presentarlo así.

**Error de calibración esperado (ECE, 10 bins):** 0.1787 en la logística y 0.1663 en el
bosque. El orden no depende del esquema de binning: ancho igual y cuantiles coinciden.

### 2.2 Dos expectativas de la literatura que no se cumplen

Se verificaron en vez de asumirse, y ambas fallan:

| Expectativa | Predicción | Observado |
|---|---|---|
| El bosque comprime las probabilidades hacia el centro | rango menor en el RF | Rango p05–p95: **RF 0.5595**, logística **0.1445** |
| La logística está bien calibrada de fábrica | sesgo pequeño | Sesgo **+0.2645** en la zona media, **+0.4000** en la alta |

La que comprime es la logística, por un factor cercano a cuatro. El factor dominante no es
el algoritmo sino el reponderado de clases, que afecta a ambos.

### 2.3 La corrección funciona, y su magnitud es el resultado

Escalado de Platt ajustado sobre bloques internos de entrenamiento (gemelos entrenados
sobre 2022, calibrador sobre 2023, decisión sobre 2024):

| Modelo | ECE antes | ECE después | Reducción | Brier antes | Brier después |
|---|---|---|---|---|---|
| Logística | 0.1748 | **0.0087** | 95.0% | 0.2204 | 0.1818 |
| Random Forest | 0.1706 | **0.0030** | 98.3% | 0.2303 | 0.1830 |

El criterio declarado exigía 20% de reducción sin empeorar el Brier; se obtuvo 95–98% y el
Brier mejoró también.

**Lectura para el informe:** que una transformación de **dos parámetros** elimine
prácticamente todo el error demuestra que el desajuste no tenía estructura. Era un
desplazamiento de prior, que en escala logit es una constante aditiva — exactamente la
forma que una sigmoide puede representar. Si el problema hubiera sido de especificación del
modelo, dos parámetros no habrían bastado.

### 2.4 El submodelo condicional: el aporte principal

**El argumento anti-fuga.** NB04 excluyó `grupo_vulnerable` y `rango_etario` porque valen
`Sin victimas` en exactamente los 46,841 casos de `solo_danos`: eran una codificación del
target, no predictores. Al restringirse a los 43,105 siniestros **con víctima**, esa
categoría no existe por construcción y las variables pasan a informar sobre *quién* fue la
víctima, no sobre si la hubo.

**No se relajó la regla: se cambió la pregunta hasta que la regla dejó de tener objeto.**

Se verificó además, contra el código de NB03, que ninguna de las dos variables se construyó
mirando el desenlace: `grupo_vulnerable` es el usuario más vulnerable presente por
prioridad fija, y `rango_etario` es el del conductor. Sin esa comprobación el argumento no
se sostendría.

**Resultado sobre el conjunto de prueba (2025–2026), usado una sola vez:**

| Esquema | AUC-PR | Lift | ROC-AUC | Precisión | Recall | F1 de `fatal` |
|---|---|---|---|---|---|---|
| Sin variables de víctima (9) | 0.1348 | 1.162 | 0.5357 | 0.1309 | 0.4365 | 0.2014 |
| **Con variables de víctima (11)** | **0.2163** | **1.865** | **0.6835** | **0.1818** | **0.6247** | **0.2816** |
| *Piso (prevalencia)* | *0.1160* | *1.000* | *0.500* | | | |

**Δ AUC-PR = +0.0815, IC 95% [+0.0689, +0.0942]** por bootstrap pareado sobre 2,000
remuestreos. El intervalo excluye el cero.

En términos operativos: detecta **1,185 de 1,897** siniestros fatales frente a 828 del
esquema reducido, y con **menos** falsos positivos (5,333 contra 5,496).

**El resultado inesperado, que conviene reportar:** sin las variables de víctima el
submodelo está **prácticamente en el azar** —AUC-PR 0.1348 sobre un piso de 0.1160,
ROC-AUC 0.5357—. El tipo de siniestro y el tipo de vehículo aportan muy poco una vez que se
condiciona sobre que hubo víctima. La severidad, dado que alguien resultó afectado, depende
sobre todo de quién fue ese alguien.

### 2.5 Interpretabilidad: SHAP valida el EDA por vía independiente

Las dos variables de víctima reúnen el **59.1%** de la atribución SHAP total, con
`grupo_vulnerable` en 46.3%.

**Concordancia con NB02, con una advertencia metodológica importante:**

| Grupo | SHAP medio | Letalidad observada |
|---|---|---|
| Peatón | +0.0991 | 0.2294 |
| Motociclista | +0.0145 | 0.1500 |
| Ciclista | +0.0108 | 0.1371 |
| Ocupante de vehículo | −0.0904 | 0.0407 |

Concordancia de rangos **exacta** en los cuatro grupos, y **Spearman 0.9560** entre
atribución media y letalidad para `rango_etario` sobre trece categorías.

**La advertencia:** NB02 mide por **víctima** (motociclistas 42.0% de los fallecidos);
`matriz_severidad` mide por **siniestro**, y ahí el orden se invierte (peatón 42.47%). No
es una contradicción sino la unidad de análisis: un siniestro con peatón y motociclista se
etiqueta «peatón» por la regla de prioridad, y un siniestro de moto aporta a menudo dos
víctimas. **Lo comparable entre ambas fuentes es el ordenamiento, no los porcentajes.**

### 2.6 Para qué sirve, y para qué no

El submodelo **no entra en el prototipo predictivo**, y no por una limitación corregible:
`grupo_vulnerable` y `rango_etario` se conocen tras identificar a las víctimas. Pedirle una
predicción anticipada es pedirle que anticipe su propio condicionante.

**Usos que el desempeño sostiene:**

1. **Política pública.** Un peatón involucrado en un siniestro tiene **5.6 veces** la
   probabilidad de morir que un ocupante de vehículo cerrado (0.2294 contra 0.0407). El
   aporte del submodelo es confirmar que esa asociación persiste controlando por el resto
   de variables.
2. **Análisis retrospectivo estratificado.** Los 712 falsos negativos —fatalidades en
   perfiles de bajo riesgo estimado— son el conjunto que merece revisión cualitativa.

**Uso a descartar explícitamente:** con precisión 0.1818, no sirve para priorizar respuesta
de emergencia caso a caso. De cada seis siniestros que señala, cinco no son fatales.

---

## 3. NB10 — Clustering espacio-temporal con ST-DBSCAN

### 3.1 El veredicto: la estructura existe, pero no a la escala necesaria

Antes de agrupar nada, el notebook determina si la capa espacial **puede** contener
concentraciones viales, dado que el conjunto es sintético calibrado.

| Medición | Ventana de comparación | Resultado |
|---|---|---|
| T1 firma del generador | — | Negativo: 6 decimales pero **0.0000%** de coordenadas repetidas |
| T2 vecino más cercano | envolvente convexa por distrito | **Positivo**: 63.2% de los distritos |
| T3 ocupación de grilla de 110 m | envolvente convexa | **Positivo**: razón 0.8853 |
| **Control** ocupación | celda de ~1.1 km | **Nulo**: razón 0.9963 |
| **Control** vecino más cercano | celda de ~1.1 km | **Nulo**: razón 0.9929 |

**Por encima de ~1 km hay estructura**: 60 de 95 distritos muestran agrupamiento
significativo. **Por debajo no hay nada**: condicionando a esa intensidad gruesa, la
discrepancia máxima cae al **0.71%**, siete veces por debajo del umbral del 5% declarado.

Y la estructura gruesa no aporta: es el reparto por exposición del generador (población
censal, urbanización, AMSS, parque vehicular), documentado en NB01 y NB03, que el proyecto
ya explota mejor como término de offset desde NB05. Describe dónde hay más tránsito, no
dónde la vía es peligrosa.

**Veredicto: modo demostrativo.** ST-DBSCAN se aplica, verifica y parametriza con rigor;
sus agrupamientos no se interpretan como concentraciones viales.

### 3.2 El índice de Clark-Evans habría producido un falso positivo

Es el instrumento habitual y el notebook demuestra que aquí no sirve, **sin usar una sola
coordenada real**: construye 103 distritos artificiales con los conteos reales y reparte
los puntos de forma perfectamente uniforme dentro de cada uno.

```
Clark-Evans NACIONAL sobre datos SIN estructura alguna:  0.7929
```

La causa es el confundido entre heterogeneidad y agrupamiento: el índice supone un proceso
homogéneo y la siniestralidad no lo es (un distrito tiene 11,058 siniestros y otro 3). Un
criterio del tipo «si el índice baja de 0.9 hay estructura» habría llevado a presentar
artefactos como puntos negros.

Por eso el veredicto se apoya en **envolventes de Monte Carlo**: los sesgos de área, borde y
forma afectan por igual a lo observado y a lo simulado, y se cancelan al comparar.

### 3.3 Tres confirmaciones independientes del veredicto

**La curva de k-distancia no tiene codo.** Percentiles de la distancia al cuarto vecino:
168 m (p50), 503 m (p90), 1,385 m (p99). La máxima curvatura aparece en el percentil 98.6 y
es solo **10.6 veces** la media; un codo genuino da cocientes de varios órdenes de magnitud.

**La transición es abrupta: no hay régimen intermedio.**

| `eps1` | Grupos | Ruido | Mayor grupo |
|---|---|---|---|
| 100–500 m | **0** | 100.0% | 0 |
| 750 m | 25 | 99.1% | 75 |
| 1000 m | 73 | 90.8% | **2,538** |

Es **percolación**, la firma de un proceso de puntos homogéneo: se pasa de todo-ruido a un
componente gigante sin atravesar una zona de agrupamientos bien definidos.

**La silueta es negativa en las cinco configuraciones admisibles** (de −0.1662 a −0.6768).
Solo 5 de 120 combinaciones resultan admisibles y todas dan coeficiente negativo: cada
punto está en promedio más cerca de otro grupo que del suyo.

**Y los agrupamientos miden lo que mide el parámetro:** radio mínimo 913 m y mediano
1,174 m, frente a un `eps1` de 1,000 m. Su extensión la fija el parámetro, no el dato.

### 3.4 Decisiones técnicas defendibles

| Decisión | Justificación |
|---|---|
| Implementación propia, no `DBSCAN` con métrica combinada | Birant y Kut definen la vecindad como **conjunción** de dos umbrales (un cilindro); una métrica combinada define una esfera. Es otro algoritmo |
| Verificación doble | Idéntica a la referencia por fuerza bruta, e **índice de Rand ajustado 1.000000** contra `sklearn.DBSCAN` sobre puntos núcleo cuando `eps2` deja de atar |
| Grafo por ventana temporal deslizante | 89,946 puntos son 60.3 GB si se precomputara la matriz. La conjunción permite descartar pares sin calcularlos: 104,008 aristas en 1.4 s. **Exacto, no aproximado** |
| Haversine y no euclidiana sobre grados | NB03 usó euclidiana correctamente porque solo necesitaba un *ordenamiento*; aquí `eps1` es un radio absoluto y la anisotropía es del **2.9%** |
| Silueta sobre distancia de Chebyshev normalizada | `max(d_esp/eps1, d_tmp/eps2)`, cuya bola unitaria es exactamente la vecindad del algoritmo |

### 3.5 Contraste con el modelo de frecuencia de NB07

| Comparación | Spearman |
|---|---|
| Riesgo predicho por el XGBoost contra observado | **0.9708** |
| Riesgo predicho contra nº de agrupamientos por distrito | 0.3894 (p = 4.8 × 10⁻⁵) |

Significativo y sustantivamente débil. La lectura correcta no es «coinciden a medias»: los
agrupamientos aparecen donde hay muchos puntos, que es donde el modelo predice más
siniestros. La correlación la induce el conteo, no un acuerdo entre dos maneras de
identificar riesgo.

### 3.6 Qué haría falta para un análisis sustantivo

La conclusión **no** es que ST-DBSCAN sea inadecuado: es que el conjunto no tiene la
resolución espacial que el algoritmo necesita. Harían falta coordenadas reales
geocodificadas contra la red vial, con la repetición característica de los puntos negros.
Con un conjunto así el procedimiento se aplica **sin cambios**.

La infraestructura queda construida y verificada. Lo que falta es el dato.

---

## 4. Rigor metodológico común a ambos notebooks

Conviene que el informe lo presente como una práctica del proyecto y no como un detalle de
implementación:

| Práctica | NB09 | NB10 |
|---|---|---|
| Criterios declarados **antes** de calcular, con marca de tiempo | D1–D8 en `nb09_decisiones.json` | D1–D8 en `nb10_decisiones.json` |
| Integridad de artefactos previos por SHA-256 al abrir y al cerrar | 19 archivos intactos | 18 archivos intactos |
| Reproducción de líneas base como puerta de entrada | Métricas de NB04 al sexto decimal | — |
| Verificación de la implementación contra referencia externa | — | ARI 1.000000 contra `sklearn.DBSCAN` |
| Conjunto de prueba usado una sola vez | Sí | No aplica (no hay selección supervisada) |
| Intervalos de confianza antes de declarar un efecto | Bootstrap pareado, 2,000 remuestreos | Envolventes de Monte Carlo, 199 simulaciones |
| Correcciones de criterio declaradas explícitamente | — | Dos, marcadas `corregido_post_observacion` |

**Sobre las correcciones de NB10.** El notebook documenta que su criterio se corrigió dos
veces tras observar los datos: T1 pasó de disyunción a conjunción, y el veredicto pasó de
binario a escalar cuando dos pruebas del mismo fenómeno resultaron contradictorias. Ambas
quedan registradas. Declararlas es preferible a ocultarlas, y la segunda condujo a un
resultado más preciso que el criterio original podía representar.

---

## 5. Limitaciones a declarar en el informe

**NB09**

- La corrección de calibración se demostró sobre gemelos entrenados con un tercio de los
  datos: queda establecido que el **procedimiento** funciona, no que los artefactos
  publicados de NB04 estén calibrados.
- El submodelo sobreestima el nivel en prueba (media predicha 0.1391 contra prevalencia
  0.1160), porque la letalidad descendió de 0.1526 en 2022 a 0.1140 en 2025. Es
  desplazamiento temporal, del mismo tipo que NB05 corrigió para la frecuencia.
- Precisión de 0.1818: inutilizable para decisiones caso a caso.
- Variables ex post: no admite uso predictivo, y no es corregible.
- 712 fatalidades sin explicar por las variables disponibles.

**NB10**

- Los agrupamientos no son concentraciones viales. Es la limitación principal y condiciona
  todo el notebook.
- La ventana de simulación se derivó de los propios puntos (no hay geometrías de distrito
  en el repositorio), lo que sesga T2 y T3 hacia detectar agrupamiento. Por eso el control
  de escala es el decisorio.
- El umbral de efecto del 5% es una elección; el efecto medido está siete veces por debajo.
- `eps2` en días: una estructura sub-diaria quedaría fuera de alcance.

**Común a ambos**

- El conjunto es sintético, calibrado en NB01 sobre agregados de ONASEVI/FONAT. Todo lo
  anterior describe el conjunto, no la siniestralidad salvadoreña real. La validación
  externa exige registros reales.

---

## 6. Artefactos generados

**NB09** — `notebooks/NB09_calibracion_y_submodelo_severidad.ipynb` (43 celdas, 19 de
código, 3 figuras)

| Tipo | Archivos |
|---|---|
| Modelos | `severidad_condicional_nb09.pkl`, `severidad_condicional_sin_victima_nb09.pkl`, dos calibradores |
| Resultados | `nb09_metricas.json`, `nb09_especificacion.json`, `nb09_decisiones.json`, `nb09_hashes_iniciales.json` |
| Figuras | `nb09_01_fiabilidad_3clases.png`, `nb09_02_submodelo_calibracion_pr.png`, `nb09_03_shap_submodelo.png` |

**NB10** — `notebooks/NB10_clustering_espacio_temporal.ipynb` (36 celdas, 18 de código,
4 figuras)

| Tipo | Archivos |
|---|---|
| Datos | `nb10_asignacion_clusters.csv` (5.3 MB) |
| Resultados | `nb10_parametros.json`, `nb10_clusters.json`, `nb10_estructura_espacial.json`, `nb10_decisiones.json`, `nb10_hashes_iniciales.json` |
| Figuras | `nb10_01_estructura_espacial.png`, `nb10_02_curva_k_distancia.png`, `nb10_03_barrido_parametros.png`, `nb10_04_clusters_y_contraste.png` |

Ningún notebook modificó artefactos previos, `src/config.py` ni añadió dependencias.
Ninguno de los archivos generados supera los 100 MB.

---

## 7. Figuras recomendadas para el informe

Si el informe admite un número limitado de figuras, estas cuatro sostienen el argumento:

| Figura | Qué muestra |
|---|---|
| `nb09_01_fiabilidad_3clases.png` | Los diagramas de fiabilidad y la ocupación de cada bin: hace visible el desplazamiento hacia la uniforme |
| `nb09_02_submodelo_calibracion_pr.png` | La curva precisión-recall con y sin variables de víctima: es el aporte del notebook en una imagen |
| `nb10_01_estructura_espacial.png` | La discrepancia contra dos ventanas: muestra cómo desaparece al condicionar a la escala gruesa |
| `nb10_03_barrido_parametros.png` | La transición de percolación al variar `eps1` |
