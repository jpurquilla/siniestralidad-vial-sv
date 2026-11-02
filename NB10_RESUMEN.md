# NB10 — Resumen para la defensa

**Notebook:** `notebooks/NB10_clustering_espacio_temporal.ipynb`
**Rama:** `feat/nb10` · **Semilla:** 42 · **Datos:** `data/raw/siniestros.csv` (89,946 siniestros, 103 distritos)

---

## 1. Qué responde este notebook

El anteproyecto compromete ST-DBSCAN en su marco teórico, citando a Birant y Kut (2007) y
el coeficiente de silueta. En la Etapa 1 no se ejecutó porque no estaba en aquella rúbrica.
NB10 salda ese compromiso.

Pero antes de agrupar nada, determina si la capa espacial **puede** contener
concentraciones viales. Los datos son sintéticos calibrados: reproducen los agregados
oficiales de ONASEVI/FONAT, pero las coordenadas individuales fueron generadas repartiendo
siniestros entre distritos por exposición. Si dentro de cada distrito se sortearon de forma
uniforme, cualquier «punto negro» que el algoritmo encuentre es un artefacto.

**Resultado: la estructura existe, pero no a la escala que ST-DBSCAN necesita.**

---

## 2. El hallazgo central, en una tabla

| Medición | Ventana de comparación | Resultado |
|---|---|---|
| T1 firma del generador | — | Negativo: 6 decimales pero **0.0000%** de coordenadas repetidas |
| T2 vecino más cercano | envolvente convexa por distrito | **Positivo**: 63.2% de los distritos |
| T3 ocupación de grilla de 110 m | envolvente convexa | **Positivo**: razón 0.8853 |
| **Control** ocupación | celda de ~1.1 km | **Nulo**: razón 0.9963 |
| **Control** vecino más cercano | celda de ~1.1 km | **Nulo**: razón 0.9929 |

**Por encima de ~1 km hay estructura.** Los siniestros no se reparten uniformemente dentro
del distrito: 60 de 95 distritos muestran agrupamiento significativo, con razones de hasta
0.7464 en San Salvador.

**Por debajo de ~1 km no hay nada.** Condicionando a esa intensidad gruesa —reubicando cada
punto dentro de su propia celda de 1.1 km— la discrepancia máxima cae al **0.71%**, siete
veces por debajo del umbral del 5% declarado.

**Y la estructura gruesa no sirve**, porque es el reparto por exposición que el generador
aplicó (población censal, urbanización, AMSS, parque vehicular), documentado en NB01 y
NB03. El proyecto ya la explota por una vía mejor: es exactamente lo que el término de
offset de los modelos de frecuencia incorpora desde NB05. Describe dónde hay más tránsito,
no dónde la vía es peligrosa.

**Veredicto:** `MODO = "demostrativo"`. ST-DBSCAN se aplica, se verifica y se parametriza
con rigor; sus agrupamientos no se interpretan como concentraciones viales.

---

## 3. Por qué el índice de Clark-Evans no se usó como criterio

Es el instrumento habitual y **habría dado un falso positivo**. El notebook lo demuestra
antes de aplicarlo, sin tocar una sola coordenada real: construye 103 distritos
artificiales con los conteos reales y reparte los puntos dentro de cada uno de forma
perfectamente uniforme.

```
Clark-Evans NACIONAL sobre datos SIN estructura alguna: 0.7929
```

La razón es el confundido entre heterogeneidad y agrupamiento. El índice supone un proceso
homogéneo, y la siniestralidad no lo es: un distrito tiene 11,058 siniestros y otro 3. Los
densos aportan multitud de distancias cortas y el índice cae por debajo de 1 aunque dentro
de cada distrito los puntos estén perfectamente dispersos.

Si el criterio hubiera sido «si Clark-Evans baja de 0.9, hay estructura», el notebook
habría concluido que la hay y habría presentado artefactos como puntos negros.

Por eso D2 apoya el veredicto en **envolventes de Monte Carlo**: los sesgos de área, borde
y forma afectan por igual a lo observado y a lo simulado, y se cancelan al comparar.

---

## 4. Tres resultados que confirman el veredicto por otra vía

### 4.1 La curva de k-distancia no tiene codo

| Percentil | Distancia al 4º vecino |
|---|---|
| p50 | 168 m |
| p90 | 503 m |
| p99 | 1,385 m |

La máxima curvatura aparece en el percentil 98.6 y es solo **10.6 veces** la media. Un codo
genuino da cocientes de varios órdenes de magnitud. Sin dos regímenes de densidad no hay
codo, que es lo que la Sección 2 predijo.

### 4.2 La transición es abrupta: no hay régimen intermedio

| `eps1` | Grupos | Ruido | Mayor grupo |
|---|---|---|---|
| 100–500 m | **0** | 100.0% | 0 |
| 750 m | 25 | 99.1% | 75 |
| 1000 m | 73 | 90.8% | **2,538** |

Hasta los 500 metros el algoritmo no encuentra nada. Entre 750 y 1000 aparecen grupos y de
inmediato uno crece a 2,538 puntos. Es **percolación**: la firma de un proceso homogéneo.
Un conjunto con puntos negros reales mostraría grupos estables sobre un rango amplio de
`eps1`, porque la estructura tendría escala propia.

### 4.3 La silueta es negativa en las cinco configuraciones admisibles

| `eps1` | `eps2` | `min_samples` | Grupos | Ruido | Silueta |
|---|---|---|---|---|---|
| 1000 m | 14 d | 20 | 73 | 90.8% | **−0.1662** |
| 750 m | 30 d | 20 | 68 | 86.4% | −0.4696 |
| 1000 m | 30 d | 20 | 57 | 69.0% | −0.4894 |
| 1000 m | 30 d | 10 | 190 | 53.3% | −0.5693 |
| 750 m | 30 d | 10 | 149 | 64.8% | −0.6768 |

Solo 5 de 120 combinaciones son admisibles y **las cinco dan silueta negativa**: cada punto
está en promedio más cerca de otro grupo que del suyo. No es agrupamiento mediocre; es
ausencia de agrupamiento con etiquetas encima.

### 4.4 Los agrupamientos miden lo que mide el parámetro

Radio mínimo 913 m, mediano 1,174 m — frente a un `eps1` de 1,000 m. **Su extensión la fija
el parámetro, no el dato.** Una concentración vial real tendría forma alargada y tamaño
propio.

---

## 5. Decisiones técnicas y su justificación

### 5.1 Implementación propia, no `DBSCAN` con métrica combinada

Birant y Kut definen la vecindad como **conjunción** de dos umbrales: un cilindro en el
espacio-tiempo. Una métrica combinada define una esfera, que admite puntos lejanos en el
espacio si están próximos en el tiempo. **Es otro algoritmo**, y presentarlo como ST-DBSCAN
sería inexacto ante un jurado que haya leído el artículo.

Se implementó el predicado y se reutilizó la expansión de DBSCAN, que sí es idéntica.

**Verificada dos veces:** idéntica a la referencia por fuerza bruta, y con índice de Rand
ajustado de **1.000000** contra `sklearn.DBSCAN` sobre los puntos núcleo cuando `eps2` deja
de atar. La comparación se hace sobre núcleos porque los puntos frontera son ambiguos por
definición en DBSCAN —asignables a más de un grupo según el orden de recorrido—, propiedad
del algoritmo original que scikit-learn también documenta.

### 5.2 El problema de escala, resuelto sin aproximar

89,946 puntos son 8.1 × 10⁹ pares: **60.3 GB** si se precomputara la matriz.

La salida está en la propia conjunción: dos puntos separados por más de `eps2` días no
pueden ser vecinos. Se ordena por fecha, se recorre con una ventana deslizante de ±`eps2`
días y se calcula Haversine solo dentro de ella. El grafo resultante es **exacto**, no
aproximado. Con `eps2` de 7 días son 104,008 aristas en 1.4 segundos.

### 5.3 Haversine, y por qué difiere de NB03

NB03 usó euclidiana sobre grados para asignar estaciones meteorológicas. **Aquella decisión
era correcta**: solo necesitaba un *ordenamiento* de cercanía, que la euclidiana preserva.

Aquí `eps1` es un **radio absoluto** comparado contra un umbral, y a la latitud del país un
grado de latitud mide 1,112 m por cada 0.01° mientras uno de longitud mide 1,080: una
anisotropía del **2.9%** que mueve puntos dentro o fuera de la vecindad.

Se descartó UTM porque el país cruza el límite entre las zonas 15N y 16N.

### 5.4 El coeficiente de silueta: tres decisiones que el anteproyecto no fija

- **Distancia:** Chebyshev normalizada `max(d_esp/eps1, d_tmp/eps2)`, cuya bola unitaria es
  exactamente la vecindad de ST-DBSCAN. Evaluar con otra métrica daría un número difícil de
  interpretar.
- **Ruido:** excluido. Se reporta también incluido: pasa de −0.1662 a **−0.6667**, porque el
  ruido reúne el 90.8% de los puntos repartidos por todo el país.
- **Advertencia declarada:** la silueta penaliza formas alargadas, que es justamente la
  forma que tendría una concentración vial real. Por eso no se usa sola.

---

## 6. Dos correcciones de criterio, declaradas

Ambas figuran en `nb10_decisiones.json` con la marca `corregido_post_observacion`.

**Primera — T1, de disyunción a conjunción.** Se declaró «precisión truncada **o**
repetición masiva». Las coordenadas tienen exactamente 6 decimales, así que bajo lectura
literal T1 daba positivo. Pero 6 decimales son 11 cm: redondear es cosmético y no produce
colisiones entre sorteos continuos, y en efecto hay **0 repetidas en 89,946**. La firma de
un geocodificador son las dos marcas **juntas**.

**Segunda — D3, de binario a escalar.** El criterio original era una marca binaria
gobernada por T1 y T2 contra la envolvente convexa. Al medir, T2 dio positivo mientras el
control daba nulo: dos pruebas del mismo fenómeno en direcciones opuestas. La contradicción
no estaba en los datos sino en el criterio, que trataba como una sola cosa lo que son dos
escalas. Un veredicto binario no puede representar «hay estructura a un kilómetro y no la
hay a doscientos metros», que es lo que el conjunto muestra.

**Un error que corregí durante el desarrollo y conviene que sepas.** En una prueba
preliminar calculé la significación de T2 con el signo invertido: marqué como agrupados los
distritos que en realidad estaban *dispersos*. Eso dio 7.4% en lugar del 63.2% correcto, y
sobre esa cifra equivocada propuse un veredicto. El error se detectó al ejecutar el
notebook, cuyo código usaba el criterio correcto. La discrepancia entre ambos resultados es
lo que llevó a revisar y a rediseñar D3.

---

## 7. Contraste con NB07

| Comparación | Spearman |
|---|---|
| Riesgo predicho por el XGBoost contra observado | **0.9708** |
| Riesgo predicho contra nº de agrupamientos por distrito | 0.3894 (p = 4.8 × 10⁻⁵) |
| Riesgo predicho contra proporción agrupada | 0.3860 |

Significativo y sustantivamente débil. La lectura no es «coinciden a medias»: es que los
agrupamientos aparecen donde hay muchos puntos, que es donde el modelo predice más
siniestros. La correlación que existe la induce el conteo, no un acuerdo entre dos maneras
de identificar riesgo.

---

## 8. Limitaciones

| Limitación | Evidencia | Implicación |
|---|---|---|
| Los agrupamientos no son concentraciones viales | Sin estructura bajo ~1 km; efecto 0.71% | Ninguno señala dónde intervenir. Condiciona todo el notebook |
| La ventana de simulación se derivó de los puntos | No hay geometrías de distrito en el repositorio | La envolvente convexa incluye zonas sin vías y sesga T2/T3 hacia detectar agrupamiento. Por eso decide el control |
| El umbral del 5% es una elección | Con 89,946 puntos todo es significativo | Otro umbral podría cambiar el veredicto. El efecto medido está 7 veces por debajo |
| La silueta penaliza formas alargadas | Sección 4.2 | Su signo negativo es consistente con la ausencia de estructura, pero no la demuestra solo |
| `eps2` en días | El conjunto tiene resolución de minuto | Una estructura sub-diaria quedaría fuera de alcance |
| Conjunto sintético | Calibrado en NB01 | Todo describe el conjunto, no la siniestralidad real |

---

## 9. Qué haría falta para un análisis sustantivo

La conclusión **no** es que ST-DBSCAN sea inadecuado para el problema: es que el conjunto
no tiene la resolución espacial que el algoritmo necesita. La distinción importa porque
señala qué conseguir.

Harían falta coordenadas reales geocodificadas contra la red vial, con la repetición
característica de los puntos negros: la misma intersección apareciendo decenas de veces.
Con un conjunto así, el procedimiento de este notebook se aplica **sin cambios**: la
Sección 2 daría estructura fina, el modo pasaría a sustantivo, y las secciones 4 y 5
producirían resultados interpretables.

La infraestructura queda construida y verificada. Lo que falta es el dato.

---

## 10. Archivos

| Archivo | Contenido |
|---|---|
| `notebooks/NB10_clustering_espacio_temporal.ipynb` | 36 celdas, 18 de código, 4 figuras |
| `reports/results/nb10_parametros.json` | Grilla completa del barrido con métricas |
| `reports/results/nb10_clusters.json` | Caracterización de los 73 agrupamientos |
| `reports/results/nb10_estructura_espacial.json` | Veredicto por escala y las tres pruebas |
| `reports/results/nb10_decisiones.json` | D1–D8 con las dos correcciones marcadas |
| `reports/results/nb10_hashes_iniciales.json` | Referencia de integridad |
| `data/processed/nb10_asignacion_clusters.csv` | Cluster por siniestro (~5 MB, versionable) |
| `reports/figures/nb10_01_estructura_espacial.png` | Ocupación observada contra dos ventanas |
| `reports/figures/nb10_02_curva_k_distancia.png` | Curva de k-distancia y detalle de la cola |
| `reports/figures/nb10_03_barrido_parametros.png` | Grupos, ruido y percolación frente a `eps1` |
| `reports/figures/nb10_04_clusters_y_contraste.png` | Mapa de agrupamientos y contraste con NB07 |
| `NB10_RESUMEN.md` | Este documento |

**Nada previo fue modificado.** La celda 1.1 registra el SHA-256 de 18 artefactos de NB01 a
NB09 y la 6.2 los vuelve a verificar. No se tocó `src/config.py` ni se añadió ninguna
dependencia: ST-DBSCAN se implementó sobre `numpy` y `scipy`.
