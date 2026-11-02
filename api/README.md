# API de predicción de siniestralidad vial — El Salvador

Backend del prototipo de la Fase 6 de la tesina. Expone por HTTP el modelo
final de frecuencia de NB12 (XGBoost con offset, conteo por distrito × día ×
franja horaria, 103 distritos). Este documento asume que nunca usaste FastAPI.

---

## 1. Qué es y qué no es

- Sirve **predicciones del modelo ya entrenado**. No reentrena nada, no
  modifica ningún notebook, artefacto de `models/` ni archivo de
  `reports/` — solo los lee.
- Los datos son **sintéticos**, calibrados sobre agregados de ONASEVI/FONAT
  (ver `NB01`). Toda respuesta de predicción lo declara en el campo `aviso`.

### Modelo servido

El modelo vigente es el **modelo final de NB12** (`modelo_final = "ampliado"`
en `reports/results/nb12_especificacion.json`): `frecuencia_xgboost_offset_v3.pkl`.
El archivo, las columnas y `log_tasa_base` se leen de esa especificación al
arrancar; el nombre del archivo cargado se informa en `GET /health` y en
`GET /api/v1/modelo/info` (`version_modelo`). El modelo de NB07 aparece solo
como referencia histórica (`metricas_referencia_nb07`), nunca como vigente.

Las 12 variables, en el orden del modelo:

| # | Variable | Origen |
|---|---|---|
| 1-5 | `es_lluviosa`, `es_finde`, `dia_semana`, `periodo_agostino`, `es_feriado` | calendario (fecha) |
| 6 | `prcp_mensual` | normal climatológica del distrito y mes, o el valor enviado |
| 7-8 | `pct_urbano`, `poblacion_baja` | censo 2024 (constantes por distrito) |
| 9 | `franja_cod` | franja pedida: 0 Madrugada, 1 Manana, 2 Tarde, 3 Noche |
| 10 | `crec_parque_12m` | parque vehicular, por distrito y mes (ver `fuente_parque`) |
| 11-12 | `pct_hogares_auto`, `pct_trabaja_otro_lugar` | censo 2024 (constantes por distrito) |

El offset es `log_exposicion_v2 + log_tasa_base`. Como `franja_cod` es una
variable del modelo, **cada franja predice un valor distinto** (la madrugada
es la menor).

### Datos base: matriz v3 comprimida

Los catálogos (distritos, clima, exposición, parque, comparación histórica)
salen de `data/processed/matriz_frecuencia_v3.csv.gz` (676,504 filas,
2022-01 a 2026-06, versionada **comprimida**; pandas y `gzip` la leen
directamente). La API no depende de `matriz_frecuencia_v2.csv`, que no está
versionada. Los nombres con tilde de distritos y departamentos salen del
censo (`data/raw/TAB_POB_AREA_1.xlsx`), cruzados por `distrito_key`; el
`codigo` (sin tildes) no cambia.

### `fuente_parque`

Cada predicción trae `fuente_parque`: `"observado"` si `crec_parque_12m` es el
del mes consultado (2022-01 a 2026-06), o `"ultimo_disponible"` si la fecha
cae fuera de ese rango y se usó el mes disponible más cercano del distrito.

### Calibración

En todo el proyecto, **calibración = predicho / observado**: 1.0 es perfecta,
mayor que 1 es sobreestimación. Ej.: rural ≈ 1.20 en el conjunto de prueba
significa que el modelo predice un 20% de más.

---

## 2. Instalación

Requiere **Python 3.11 o 3.12**. Desde la carpeta `api/`:

```bash
# 1. Crear un entorno virtual PROPIO de la API (no el de la raíz del repo)
python3.12 -m venv .venv

# 2. Activarlo
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

**macOS:** XGBoost necesita OpenMP en el sistema (no lo instala pip):

```bash
brew install libomp
```

Si `import xgboost` falla con un error sobre `libxgboost.dylib`, es por esto.

---

## 3. Variables de entorno

Copiar `.env.example` a `.env` y ajustar si hace falta:

```bash
cp .env.example .env
```

Todas las rutas tienen un default razonable relativo a la raíz del
repositorio (el padre de `api/`), así que **con el repositorio tal como está
clonado, no hace falta configurar nada** para arrancar en local. Ver la
tabla completa de variables en `.env.example` y en `app/config.py`.

---

## 4. Arrancar el servicio

```bash
uvicorn app.main:app --reload --port 8000
```

- **Documentación interactiva (Swagger UI):** http://localhost:8000/docs
- **Documentación alternativa (ReDoc):** http://localhost:8000/redoc
- **Chequeo de salud:** http://localhost:8000/health

Si el arranque falla con un mensaje sobre `log_tasa_base`, un archivo que no
se encuentra, o "features" del modelo — es intencional. Ver la sección 6.

---

## 5. Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Estado del servicio: modelo cargado, `version_modelo` (archivo cargado), `log_tasa_base` y resultado de la predicción canaria |
| GET | `/api/v1/modelo/info` | Modelo final (NB12): `version_modelo`, `fecha_entrenamiento`, 12 `variables`, `offset`, `metricas`, `metricas_referencia_nb07`, `pruebas_significancia` (t pareada mensual y Diebold-Mariano), `importancia_shap`, `calibracion_por_franja` y 6 `limitaciones`. Todo número se lee de `nb12_metricas.json` |
| GET | `/api/v1/distritos` | Los 103 distritos con código, nombre (con tildes), `pct_urbano`, población y variables censales |
| GET | `/api/v1/distritos/geojson` | GeoJSON para el mapa (ver sección 7 si falta) |
| POST | `/api/v1/predicciones` | Predicción para uno o varios distritos, rango de fechas y franjas (con `fuente_prcp` y `fuente_parque`). Una sola llamada vectorizada al modelo: un mes completo de los 103 distritos (12,360 celdas) tarda ~0.15 s |
| GET | `/api/v1/ranking` | Distritos ordenados por siniestros esperados en un rango y franjas, con su tercil |
| GET | `/api/v1/impacto-ley-cero-tolerancia` | Serie anual 2022-2026 real (NB06), `tasas_por_vehiculo` (siniestros por 1,000 y fallecidos por 10,000 vehículos, con parque real ONASEVI al 31-dic; el año parcial usa el corte de fin del último mes cubierto, p. ej. 2026-06-30 para «parcial (ene-jun)») y el impacto declarado de la ley vigente desde 2024-12-20. En `resumen`, `variacion_interanual_pct` es `null` explícito donde no hay año previo comparable. Retrospectivo, nunca predictivo |
| GET | `/api/v1/usuarios-vulnerables` | Reparto de fatales y letalidad por tipo de usuario y por rango etario (NB09 §4, retrospectivo) |
| GET | `/api/v1/comparacion-historica` | Observado real vs. predicho por el modelo, sumado por distrito sobre las 224,952 filas de la **partición de prueba** (2025-2026) — diagnóstico de ajuste, no predicción |
| GET | `/api/v1/equidad` | Calibración (predicho/observado) y Spearman predicho-vs-observado por urbano/rural (umbral de NB08 §4) y por alta/baja densidad poblacional |
| GET | `/api/v1/siniestros/puntos` | Puntos individuales (lat/lon reales) de `siniestros.csv`, con filtros server-side (fecha, tipo de siniestro, vehículo, hora, departamento/distrito, factor causante) para el mapa de hotspots estilo ONASEVI. Tope de 90,000 puntos por respuesta (413 si se excede) |
| GET | `/api/v1/siniestros/opciones-filtro` | Valores reales disponibles para poblar los filtros del mapa de hotspots (tipos de siniestro, de vehículo, factores causantes, departamentos) |

Ejemplo de `POST /api/v1/predicciones`:

```json
{
  "distritos": ["san salvador", "santa ana"],
  "fecha_inicio": "2024-06-15",
  "fecha_fin": "2024-06-16",
  "franjas": ["Tarde (12-17)", "Noche (18-23)"]
}
```

Si se omite `prcp_mensual`, la respuesta usa la normal climatológica del
distrito y mes, y lo indica en `fuente_prcp` (`"normal_climatologica"` vs
`"observada"`). `fuente_parque` indica si el crecimiento del parque es del
mes consultado o el último disponible (ver sección 1).

En `/api/v1/ranking`, cada distrito trae su `tercil` (`alto`/`medio`/`bajo`)
y `confiabilidad_spearman`. Esa agrupación es **fija** (35/34/34 distritos por
total observado en el conjunto de prueba 2025-2026, protocolo de NB07 §5.4),
calculada con el modelo final en `terciles_nb12.json`; no se recalcula según
el rango de fechas — ver `scripts/generar_terciles.py`. **Nota:** según NB08,
el Spearman por tercil depende más del ancho del tercil que del modelo; es
descriptivo, no una medida de calidad. El ranking suma las franjas pedidas
(por defecto las 4), una fila por franja con su `franja_cod`.

Los cuatro endpoints de `impacto-ley-cero-tolerancia`, `usuarios-vulnerables`,
`comparacion-historica` y `equidad` son **retrospectivos**: describen el
histórico 2022-2026 (o, en `comparacion-historica`/`equidad`, qué tan bien el
modelo ajustó ese histórico). Ninguno acepta parámetros de fecha ni predice
nada — no confundirlos con `/api/v1/predicciones`. `comparacion-historica`
corre el modelo en lote sobre `matriz_frecuencia_v3.csv.gz`, filtrando
**únicamente la partición `split == "test"`** (224,952 de las 676,504 filas,
2025-2026) antes de agregar — regla D1 de NB08
(`exigir_particion_de_diagnostico`): el conjunto de prueba es la única
fuente válida para reportar una métrica de desempeño, nunca se mezcla con
train, porque el modelo ya vio esas filas al entrenar y eso infla la
calibración/Spearman de forma artificial. Esto corre una sola vez, en el
lifespan (ver `ComparacionHistoricaAdapter`), reusando
`XGBoostFrecuenciaAdapter.predecir_lote` — nunca reimplementa la inferencia,
por la misma razón que la sección 6 explica a continuación.

`/api/v1/siniestros/puntos` también es retrospectivo (89,946 siniestros
individuales de `siniestros.csv`, 2022-2026) y replica el filtrado
server-side de observatoriovial.fonat.gob.sv. Su respuesta incluye un
`aviso` que cita el hallazgo de NB10: **no hay estructura espacial real por
debajo de ~1 km** en este conjunto (efecto medido 0.71%, muy por debajo del
umbral de significancia) — los clusters visuales reflejan reparto por
exposición poblacional, no concentraciones viales verificadas. Por eso el
resto de la API es deliberadamente coroplética (por distrito) y no de
puntos: este endpoint es la única excepción, autorizada explícitamente para
replicar el mapa de hotspots de ONASEVI/FONAT.

---

## 6. El fallo silencioso de `log_tasa_base` (léase antes de tocar el modelo)

El modelo de frecuencia (`frecuencia_xgboost_offset_v3.pkl`) predice sobre un
**offset**: `base_margin = log_exposicion_v2 + log_tasa_base`.

`log_tasa_base = -12.936335060885538` **no está guardado dentro del
`.pkl`** — vive en `reports/results/nb12_especificacion.json`. Si se omite
al construir el `base_margin`, **XGBoost no lanza ningún error**: simplemente
predice ~415,000 veces más grande. Es el tipo de bug que pasa una revisión
manual porque el código "corre bien".

Por eso este servicio:

1. **Nunca hardcodea `log_tasa_base`** en el código de la aplicación — se
   lee de `nb12_especificacion.json` en cada arranque
   (`app/infrastructure/modelo/xgboost_adapter.py`).
2. Al arrancar, compara ese valor leído contra una constante de referencia y
   valida que el modelo tenga 12 features, que las columnas de la especificación
   coincidan en nombre y orden con `FeaturesCelda.como_lista()`, y que la
   `codificacion_franja` coincida con el enum `Franja` del dominio. Si algo no coincide,
   **el servicio se niega a arrancar** con un mensaje explícito — nunca
   arranca "degradado".
3. Al arrancar, corre una **predicción canaria** sobre una celda de
   referencia (San Salvador, 2024-06-15, franja Tarde; valores leídos de la matriz v3) y verifica que caiga
   en el rango `[0.01, 20.0]`. Si no, tampoco arranca.
4. Solo un lugar del código (`XGBoostFrecuenciaAdapter.predecir_lote`) suma
   `log_tasa_base` al construir el `base_margin` real de XGBoost — por
   diseño, ni los casos de uso ni los routers pueden olvidarlo, porque
   `FeaturesCelda` (el tipo que manejan) ni siquiera tiene ese campo.
5. `tests/test_modelo_adapter.py::TestRegresionOffset` es la prueba de
   regresión más importante del proyecto: reproduce el bug a propósito
   (construyendo el `base_margin` a mano, sin `log_tasa_base`) y confirma
   que la predicción se dispara ~415,125× — el mismo número que reporta
   `NB07_RESUMEN.md`.

---

## 7. GeoJSON de distritos

Ya existe un GeoJSON real en `data/geo/distritos.geojson`: los 266 municipios
(ADM2) de El Salvador según GADM v4.1, de los cuales 103 están etiquetados
con `distrito_key` (cruzando con `GET /api/v1/distritos`); el resto se
sirven igual, sin ese campo, y el frontend los pinta como "sin dato" — así
el mapa no queda con huecos donde no hay modelo. Si el archivo llegara a
faltar o apuntar a otra ruta inválida, `GET /api/v1/distritos/geojson`
devuelve **503** con un mensaje explícito; este servicio nunca inventa
polígonos de ejemplo.

`RUTA_GEOJSON_DISTRITOS` (variable de entorno) apunta a otra ubicación si
hace falta.

---

## 8. Tests

```bash
pytest
```

Los tests se organizan así (`pytest` corre toda la suite):

- `test_modelo_adapter.py` — arranque, validaciones, y la regresión del
  offset (sección 6).
- `test_exposicion_adapter.py`, `test_calendario.py`,
  `test_distrito_repository.py`, `test_tercil_repository.py`,
  `test_geojson_adapter.py` — cada adaptador de infraestructura contra los
  artefactos reales del repositorio.
- `test_impacto_ley_adapter.py`, `test_usuarios_vulnerables_adapter.py` —
  los dos adaptadores retrospectivos, verificados reproduciendo a mano los
  números ya publicados en NB06/NB09_RESUMEN.md antes de escribir el código.
- `test_comparacion_historica_adapter.py` — agregación con un predictor
  falso (rápido) más un test de integración con el modelo real que verifica
  que la razón predicho/observado quede en un rango sano (la misma clase de
  regresión de offset que la sección 6, aplicada a este agregado).
- `test_parque_adapter.py` — `crec_parque_12m` dentro y fuera de 2022-01..2026-06.
- `test_calcular_equidad.py` — el caso de uso de equidad (calibración
  predicho/observado), con dobles en memoria.
- `test_siniestro_punto_adapter.py` — el adaptador de puntos individuales
  (`siniestros.csv`), verificando conteos reales exactos (89,946 puntos,
  1,656 "Volcamiento", 14 departamentos) y cada filtro por separado.
- `test_casos_uso.py` — orquestación de `application/` con dobles de
  prueba en memoria (no toca el modelo real).
- `test_api.py` — end-to-end con `TestClient`, incluida la validación de
  entradas inválidas (distrito inexistente, fecha absurda, franja fuera de
  rango, rango de fechas demasiado largo) y los 6 endpoints retrospectivos
  (impacto de ley, usuarios vulnerables, comparación histórica, equidad,
  puntos de siniestros y sus opciones de filtro).

Los terciles se regeneran (no corre en los tests ni en el arranque) con:

```bash
python scripts/generar_terciles.py
```

Escribe `app/infrastructure/modelo/terciles_nb12.json` con el modelo final;
no toca `terciles_nb07.json` (referencia histórica).

---

## 9. Arquitectura

Hexagonal: `domain/` y `application/` son Python puro (sin imports de
FastAPI, pandas ni xgboost); los frameworks y el I/O viven solo en
`infrastructure/`. Inyección por constructor en todas las capas — nunca
`@Inject` ni variables globales; el modelo y sus adaptadores se cargan y
validan una sola vez en el `lifespan` de `app/main.py`, nunca por request.

```
api/
  app/
    domain/            # modelos, puertos (Protocol), reglas de calendario puras
    application/        # casos de uso: predecir_celda, ranking_distritos
    infrastructure/
      modelo/          # XGBoost (NB12), exposición (NB05), parque, calendario/feriados, clima, terciles
      geo/             # catálogo de distritos, GeoJSON
      web/             # routers FastAPI, schemas pydantic, dependencias
    config.py
    main.py
  scripts/
    generar_terciles.py  # precomputa terciles_nb12.json (ver sección 5)
  tests/
```

## 10. Limitaciones del modelo

La lista completa y vigente se sirve en `GET /api/v1/modelo/info`
(`limitaciones`, 6 entradas). En resumen:

- El conjunto de datos es sintético, calibrado sobre agregados de
  ONASEVI/FONAT; las reglas con que se generó no están documentadas.
- Las variables territoriales (hogares con automóvil, movilidad laboral,
  crecimiento del parque) son asociaciones dentro del conjunto, no efectos
  causales verificados.
- En el periodo de evaluación (2025–2026) el modelo sobreestima un 20% los
  siniestros en los distritos rurales (menos de 30% de población urbana) y un
  40% en la madrugada (00:00 a 05:59).
- El ordenamiento de los distritos por riesgo es algo menos preciso que el del
  modelo anterior (correlación de Spearman de 0.945 frente a 0.971).
- Para fechas fuera del periodo de enero de 2022 a junio de 2026, el
  crecimiento del parque vehicular usa el último mes disponible
  (`fuente_parque = "ultimo_disponible"`).
- El submodelo de severidad (NB09) no se expone: sus variables se conocen
  después de identificar a las víctimas y no admiten uso predictivo.

Las cifras de las limitaciones se verifican contra `nb12_metricas.json` al
arrancar: si dejaran de coincidir, la API no arranca.

Los feriados fuera de 2022-2026 se aproximan reciclando el patrón mes/día
del último año conocido (exacto para feriados de fecha fija; aproximado para
los móviles de Semana Santa).

## 11. CORS

`CORS_ORIGENES` (lista JSON) define los orígenes permitidos; por defecto
`["http://localhost:4200"]`. Se responde sin credenciales
(`allow_credentials=False`): la API no usa cookies ni sesiones.
