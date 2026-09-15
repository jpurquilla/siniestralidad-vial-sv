# BRIEF — Ensamblar NB08 (Claude Code)

**Repositorio:** `siniestralidad-vial-sv`
**Tarea:** convertir `NB08_celdas.py` en `notebooks/NB08_tercil_medio_y_validacion.ipynb`, ejecutarlo completo y verificar contra los valores de referencia de este documento.
**Rama:** `feat/nb08` — nunca trabajar en `main`.

---

## 1. Qué es esto y qué NO es

El notebook **ya fue construido y ejecutado celda por celda** en una sesión interactiva. Todos los resultados de este brief son reales y verificados. Tu tarea es de **ensamblado y verificación**, no de diseño.

**No estás resolviendo un problema abierto. El problema ya está resuelto.** Si al ejecutar obtenés un número distinto a los de la sección 5 de este brief, el fallo está en el ensamblado o en el entorno, no en el análisis.

---

## 2. Prohibiciones absolutas

Estas reglas no admiten excepción ni interpretación. Si una tarea parece requerir violarlas, detenete y preguntá.

### 2.1 Archivos que NO se tocan

```
models/frecuencia_xgboost_offset_v2.pkl      models/frecuencia_random_forest.pkl
models/frecuencia_xgboost_v2.pkl             models/frecuencia_binomial_negativa_v2.pkl
models/frecuencia_random_forest_nb07.pkl     models/frecuencia_poisson_v2.pkl
models/frecuencia_random_forest_v2.pkl       models/frecuencia_zip_v2.pkl
                                             models/frecuencia_zinb_v2.pkl
data/processed/matriz_frecuencia_v2.csv      data/processed/matriz_frecuencia.csv
reports/results/nb05_especificacion_v2.json  reports/results/nb07_especificacion.json
reports/results/nb05_metricas.json           reports/results/nb07_metricas.json
```

Solo lectura. **No sobrescribir, no regenerar, no "mejorar", no reentrenar.** El notebook verifica esto por SHA-256 al abrir y al cerrar: si algo cambió, falla y hay que restaurar desde git antes de continuar.

### 2.2 Notebooks previos

`NB01` a `NB07` están cerrados y defendidos. No se editan, no se reejecutan, no se refactorizan.

### 2.3 `src/config.py`

**No se modifica.** NB08 deriva sus rutas desde `ROOT` precisamente para no depender de constantes que quizá no existan. Si te parece que convendría agregar algo a `config.py`, anotalo como sugerencia y no lo hagas.

### 2.4 Dependencias

**Sin paquetes nuevos.** El notebook usa solo lo ya instalado: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`, `joblib`. No agregar nada a `requirements.txt`.

### 2.5 El análisis

No cambies umbrales, criterios, semillas ni definiciones. Los criterios de decisión se fijaron **antes** de ver los resultados y se guardan con marca de tiempo en `nb08_decisiones.json`, `nb08_criterios_seccion2.json` y `nb08_criterios_seccion4.json`. Alterarlos después invalida el argumento metodológico completo del notebook.

---

## 3. Contexto mínimo necesario

### 3.1 El problema que NB08 resuelve

NB07 dejó abierto que el "tercil medio" de distritos se ordenaba peor con los modelos de exposición corregida (Spearman 0.6470) que con los de Etapa 1 (0.8794). NB08 determina que **esa caída no existe**: es un artefacto de medir correlación de rangos sobre un subgrupo comprimido por construcción.

### 3.2 Protocolo heredado (no se altera)

| Elemento | Valor |
|---|---|
| Split | train 2022-01-01 a 2024-12-31 / test 2025-01-01 a 2026-06-30 |
| Partición interna | ajuste 2022-2023 (300,760 filas) / validación 2024 (150,792 filas) |
| Jerarquía de métricas | calibración → RMSE → AIC → MAE (informativo) |
| Terciles | alto 35 / medio 34 / bajo 34, por total observado |
| Semilla | 42 |
| Matriz | 676,504 filas × 24 columnas; 103 distritos × 1,642 días × 4 franjas |

### 3.3 La trampa del modelo vigente

`frecuencia_xgboost_offset_v2.pkl` **no contiene su intercepto**. Requiere:

```python
base_margin = log_exposicion_v2 + log_tasa_base
log_tasa_base = -12.936335060885538   # vive solo en nb07_especificacion.json
```

Sin ese término las predicciones salen ~415,000 veces más grandes **y no se lanza ningún error**. La función `predecir_xgb_offset()` es la **única** implementación permitida; no la dupliques.

### 3.4 La trampa de los Random Forest

Ambos bosques fueron entrenados con la columna de exposición llamada `log_exposicion`. NB05 no agregó una columna nueva: corrigió los **valores** de la existente.

| Modelo | Nombre de columna que espera | Valores que necesita |
|---|---|---|
| `frecuencia_random_forest.pkl` (Etapa 1) | `log_exposicion` | los de `log_exposicion` |
| `frecuencia_random_forest_v2.pkl` (NB05) | `log_exposicion` | los de `log_exposicion_v2` |

scikit-learn valida los **nombres** y rechaza una matriz mal armada. **No valida los valores**: intercambiarlos desvía las predicciones hasta un 39.58% sin error. Por eso la celda 2.1 verifica la calibración de cada modelo contra `nb07_metricas.json`.

### 3.5 Gemelos: por qué existen

Los `.pkl` finales entrenaron sobre 2022-2024, así que **no pueden evaluarse sobre 2024** — ahí no predicen, recitan. Un intento previo dio Spearman idénticos al sexto decimal entre dos modelos distintos (0.998166), que fue la señal del problema.

Por eso el notebook **reinstancia los candidatos** de la búsqueda de NB07: mismos hiperparámetros leídos del JSON, entrenados solo sobre 2022-2023. No es una técnica nueva; es reproducir la partición interna que NB07 ya había definido.

---

## 4. Estructura del notebook

| Sección | Celdas | Contenido |
|---|---|---|
| 1 | 1.1–1.4 | Integridad, esquema, reproducción de líneas base, decisiones D1–D5 |
| 2 | 2.1–2.6 | ¿Es real la caída? Gemelos, techos, bootstrap, robustez, veredicto |
| 3 | 3.1–3.3 | Caracterización del rango medio (modo exploratorio) |
| 4 | 4.1–4.3 | Remedio al sesgo rural: criterio, evaluación, rechazo |
| 5 | 5.1–5.3 | Fase 5: validación cruzada, sesgos, limitaciones |
| 6 | 6.1–6.2 | Confirmación en el test y cierre |

La Sección 4 se escribió **después** de la 5 durante el desarrollo. En el archivo ya está en su lugar correcto; por eso hace falta una corrida limpia para que la numeración `In[n]` siga el orden visual.

---

## 5. Valores de verificación

Ejecutá el notebook completo y comprobá estos números. **Toda discrepancia es un fallo de ensamblado, no un hallazgo.**

### 5.1 Celda 1.3 — reproducción de líneas base

18 comparaciones, todas `ok=True` con `|dif| < 1e-6`:

| Modelo | calibración | RMSE | Spearman tercil alto |
|---|---|---|---|
| NB07 — XGBoost offset | 0.966979 | 0.412055 | 0.977519 |
| NB05 — Poisson | 0.964651 | 0.429055 | 0.813642 |
| NB05 — Binomial Negativa | 0.942160 | 0.429701 | 0.818125 |

Si esto falla, **detenete**. Todo lo demás depende de que el entorno reproduzca las líneas base.

### 5.2 Celda 2.1 — gemelos y Spearman

Control de los modelos finales sobre el test:

| Modelo | calibración esperada |
|---|---|
| Etapa 1 — Random Forest | 0.819563 |
| NB05 — Random Forest | 1.005698 |

Hiperparámetros que replican los gemelos RF: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`.

Gemelo XGBoost en validación 2024: RMSE **0.387032** (NB07 publica 0.386940), calibración **0.998928** (NB07 publica 0.999789). La diferencia de 9.2e-05 es esperada y el notebook la declara — **no la "arregles"**.

Spearman sobre validación 2024 (gemelos):

| Modelo | global | bajo | medio | alto |
|---|---|---|---|---|
| XGB offset (NB07) | 0.977982 | 0.827333 | 0.777931 | 0.965474 |
| RF Etapa 1 | 0.988322 | 0.900405 | 0.801620 | 0.978080 |
| RF NB05 | 0.941540 | 0.770924 | 0.688981 | 0.919392 |

Estructura de terciles en validación 2024:

| Tercil | n | min | max | mediana | rango relativo |
|---|---|---|---|---|---|
| bajo | 34 | 0 | 58 | 27.5 | 2.1091 |
| medio | 34 | 58 | 154 | 95.5 | 1.0052 |
| alto | 35 | 159 | 2499 | 274.0 | 8.5401 |

### 5.3 Celda 2.2 — techos de ruido

| Tercil | estimador A (semanas) | estimador B (Poisson) |
|---|---|---|
| bajo | 0.916964 | 0.955896 |
| medio | 0.753612 | 0.911825 |
| alto | 0.977001 | 0.985850 |

**El modelo vigente supera el estimador A en el tercil medio (0.7779 > 0.7536).** Es un resultado esperado y el notebook lo explica: el estimador A subestima ahí porque dividir por semanas deja menos de 50 siniestros por mitad. No es un bug.

### 5.4 Celda 2.3 — techo estructural

| Tercil | exposición sola | multiplicador solo | Poisson territorial | combinaciones territoriales |
|---|---|---|---|---|
| bajo | 0.509975 | 0.726133 | 0.744936 | 26 |
| medio | 0.366957 | 0.147180 | 0.712823 | 34 |
| alto | 0.771062 | 0.350865 | 0.861545 | 34 |

### 5.5 Celda 2.4 — bootstrap pareado (el resultado central)

RF Etapa 1 − XGB offset, validación 2024:

| Tercil | diferencia | IC inf | IC sup | excluye cero |
|---|---|---|---|---|
| bajo | 0.073072 | −0.013052 | 0.196078 | No |
| **medio** | **0.023689** | **−0.222628** | **0.249744** | **No** |
| alto | 0.012606 | −0.017966 | 0.062637 | No |

Únicas comparaciones con IC que excluye el cero: **RF Etapa 1 − RF NB05** en bajo (0.129481) y alto (0.058688).

### 5.6 Celda 2.5 — robustez

Spearman del tercil medio por variante de corte:

| Variante | n medio | rango rel. | XGB offset | RF Etapa 1 | RF NB05 |
|---|---|---|---|---|---|
| corte 30 | 43 | 1.3125 | 0.822068 | 0.899328 | 0.687033 |
| protocolo | 34 | 1.0052 | 0.777931 | 0.801620 | 0.688981 |
| corte 40 | 23 | 0.5000 | 0.549320 | 0.602225 | 0.613597 |
| tasa per cápita | 34 | 6.6722 | 0.992131 | 0.969822 | 0.925357 |

Concordancia conteo vs tasa: 57 de 103 distritos (55.3%).
Kendall tau-b señala el tercil medio como el peor para los tres modelos.

### 5.7 Celda 2.6 — veredicto

- Variación por corte: **0.3500** · Variación por modelo: **0.1126**
- Veredicto: **`no_es_real`**
- `MODO_DIAG = "exploratorio"` · `HAY_REMEDIO_JUSTIFICADO = False`

### 5.8 Celda 3.2 — resolución

Pares consecutivos distinguibles en el tercil medio:

| Modelo | proporción | separación rel. mediana | CV pred/CV obs |
|---|---|---|---|
| XGB offset | 0.0606 | 0.1685 | 1.1105 |
| RF Etapa 1 | **0.0000** | 0.1428 | 1.0220 |
| RF NB05 | 0.0303 | 0.1639 | 1.5383 |

Variación entre modelos: 0.0606 · Variación entre terciles: 0.3226

### 5.9 Celda 3.3 — efecto de distrito

Estabilidad entre 2022 y 2023: bajo 0.5193 · **medio 0.6559** · alto 0.5868 · global 0.5704
Distritos con desajuste coherente: **71 de 103 (68.9%)**

Correlaciones del desajuste: `poblacion` −0.1089 · `pct_urbano` **+0.0104** · `tasa_per_capita` +0.6075

Diez mayores desajustes (ratio observado/predicho): cinquera 0.1856, aguilares 0.4630, panchimalco 0.4646, guazapa 0.4905, santo tomás 0.5214, victoria 0.5742, atiquizaya 0.6175, comasagua 0.6918, guaymango 0.7219, tacuba 0.7248.

### 5.10 Celdas 4.2–4.3 — remedio rechazado

| Ventana | factor rural | factor resto |
|---|---|---|
| ajuste 2022-2023 | **1.002170** | 0.999478 |
| validación 2024 | **1.167806** | 0.998543 |

Mejora rural obtenida: **0.001858** contra 0.05 exigida → **RECHAZADO**.
Condiciones 2, 3 y 4 se cumplen. El Spearman rural es idéntico en las tres filas (0.922334) — es control de implementación, debe cumplirse exactamente.

### 5.11 Celda 5.1 — validación cruzada

| Pliegue | meses | g anual reestimado | desvío vs global |
|---|---|---|---|
| 1 | 12 | 0.156815 | +93.50% |
| 2 | 18 | 0.242897 | +189.71% |
| 3 | 24 | 0.092799 | +17.60% |
| 4 | 30 | 0.161050 | +98.39% |

| Variante | calib. media | sd | RMSE medio | Spearman medio |
|---|---|---|---|---|
| factor reestimado (D2) | 1.082986 | 0.079064 | 0.378598 | 0.970445 |
| factor global (con fuga) | 0.999116 | 0.033354 | 0.377727 | 0.969961 |

**La variante correcta calibra peor. Es el resultado esperado y está explicado en el markdown.**

### 5.12 Celda 5.2 — sesgos

| Subgrupo | n distritos | calibración |
|---|---|---|
| rural (pct ≤ 0.3) | 19 | **0.856306** |
| población baja | 12 | **0.879025** |
| rural (pct ≤ 0.5) | 31 | 1.001787 |
| rural (pct ≤ 0.7) | 56 | 0.996826 |
| seca / lluviosa | 103 | 1.028572 / 0.973833 |
| hábil / finde | 103 | 1.000381 / 0.995705 |

El corte urbano **no es robusto**: la brecha solo aparece con umbral 0.3. Eso localiza el hallazgo, no lo invalida.

### 5.13 Celda 6.2 — confirmación

2 de 3 conclusiones coherentes. **C3 discrepa** — el notebook lo declara y explica la causa probable (el factor del test usa el modelo final, los otros dos el gemelo: comparación no limpia, falla de diseño de la celda).

---

## 6. Procedimiento

```bash
git checkout -b feat/nb08
git status                    # debe estar limpio antes de empezar

# 1. Convertir
pip show jupytext || pip install jupytext    # si no está, usar el script alternativo (§7)
jupytext --to notebook NB08_celdas.py -o notebooks/NB08_tercil_medio_y_validacion.ipynb

# 2. Ejecutar completo
cd notebooks
jupyter nbconvert --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  NB08_tercil_medio_y_validacion.ipynb

# 3. Verificar que nada previo cambió
git status                    # solo deben aparecer archivos NUEVOS
git diff --stat               # debe estar vacío para models/ y data/processed/
```

Tiempo estimado: **30 a 45 minutos**. Entrena tres gemelos (uno XGBoost, dos Random Forest sobre 300,760 filas) más ocho modelos de validación cruzada.

### Archivos nuevos esperados

```
notebooks/NB08_tercil_medio_y_validacion.ipynb
reports/results/nb08_hashes_iniciales.json
reports/results/nb08_decisiones.json
reports/results/nb08_criterios_seccion2.json
reports/results/nb08_criterios_seccion4.json
reports/results/nb08_seccion2.json
reports/results/nb08_seccion3.json
reports/results/nb08_seccion4.json
reports/results/nb08_seccion5.json
reports/results/nb08_seccion6.json
```

**Nada más.** Si aparece cualquier otro archivo modificado, revertí y avisá.

### Commit

```bash
git add notebooks/NB08_tercil_medio_y_validacion.ipynb reports/results/nb08_*.json
git commit -m "feat(nb08): add middle-tercile analysis and robust validation"
```

---

## 7. Si `jupytext` no está disponible

No lo instales. Usá este script en su lugar:

```python
# scripts/construir_nb08.py
import json, re
from pathlib import Path

RUTA_FUENTE = Path("NB08_celdas.py")
RUTA_SALIDA = Path("notebooks/NB08_tercil_medio_y_validacion.ipynb")

texto = RUTA_FUENTE.read_text(encoding="utf-8")
bloques = re.split(r"^# %%(?: \[markdown\])?\s*$", texto, flags=re.M)
marcas = re.findall(r"^# %%( \[markdown\])?\s*$", texto, flags=re.M)

celdas = []
for marca, cuerpo in zip(marcas, bloques[1:]):
    if marca:  # markdown
        lineas = [l[2:] if l.startswith("# ") else l.lstrip("#")
                  for l in cuerpo.strip("\n").split("\n")]
        celdas.append({"cell_type": "markdown", "metadata": {},
                       "source": "\n".join(lineas).strip() + "\n"})
    else:      # codigo
        celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": cuerpo.strip("\n") + "\n"})

RUTA_SALIDA.parent.mkdir(exist_ok=True)
RUTA_SALIDA.write_text(json.dumps({
    "cells": celdas,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12.13"}},
    "nbformat": 4, "nbformat_minor": 5,
}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(celdas)} celdas -> {RUTA_SALIDA}")
```

---

## 8. Qué hacer si algo falla

| Síntoma | Causa probable | Acción |
|---|---|---|
| 1.3 no reproduce al sexto decimal | Versiones de librerías distintas | **Detener.** Reportar qué métrica y cuánta diferencia |
| `log_tasa_base` no coincide | `nb07_especificacion.json` alterado | **Detener.** Restaurar desde git |
| Integridad rota al abrir/cerrar | Algún artefacto previo se modificó | **Detener.** `git checkout` de ese archivo |
| Dos modelos con Spearman idénticos en 2024 | Se usaron los `.pkl` finales en vez de los gemelos | Revisar `usar_gemelos=True` en `construir_tabla` |
| El gemelo XGBoost difiere más de 1e-3 de NB07 | Versión de xgboost distinta | Reportar la versión instalada; no ajustar el código |
| `ValueError` de nombres de columna en un RF | Se pasó `log_exposicion_v2` con el nombre equivocado | Revisar `matriz_rf()`: renombra, no selecciona |
| `KeyError` o `IndexError` sobre `_sp` / `_pi` | Variables auxiliares pisadas | Ya corregido: se leen de `ESPEC_NB07` |
| Memoria insuficiente | Los RF finales pesan >120 MB cada uno | Verificar que se hace `del` tras cada uso |

**Regla general:** ante cualquier discrepancia numérica, reportala con el valor obtenido y el esperado. No la resuelvas ajustando umbrales, semillas o criterios.

---

## 9. Después de la corrida

1. Verificar que las 19 celdas de código tienen salida y que `In[n]` sigue el orden visual.
2. Confirmar que el cierre de 6.2 imprime `Integridad de artefactos previos: verificada (15 archivos intactos)`.
3. Completar en `NB08_RESUMEN.md` los valores de C2 y C3 del test, que quedaron pendientes.
4. **No** generar figuras nuevas ni PDF de estudio: eso es una tarea aparte.

---

## 10. Contexto de la secuencia

NB08 ocupa este lugar porque el problema abierto de NB07 condicionaba todo lo que viene después. La numeración original de Etapa 2 se corre:

| Notebook | Antes | Ahora |
|---|---|---|
| NB08 | severidad condicional | **tercil medio y validación robusta** |
| NB09 | ST-DBSCAN | calibración de severidad + submodelo condicional |
| NB10 | fairness | ST-DBSCAN |
| NB11 | — | fairness urbano/rural |

Parte del contenido de NB11 ya está cubierto por la Sección 5.2 de NB08.
