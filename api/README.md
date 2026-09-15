# API de predicción de siniestralidad vial — El Salvador

Backend del prototipo de la Fase 6 de la tesina. Expone por HTTP el modelo de
frecuencia entrenado en NB07 (XGBoost, conteo por distrito × día × franja
horaria, 103 distritos). Este documento asume que nunca usaste FastAPI.

---

## 1. Qué es y qué no es

- Sirve **predicciones del modelo ya entrenado**. No reentrena nada, no
  modifica ningún notebook, artefacto de `models/` ni archivo de
  `reports/` — solo los lee.
- Los datos son **sintéticos**, calibrados sobre agregados de ONASEVI/FONAT
  (ver `NB01`). Toda respuesta de predicción lo declara en el campo `aviso`.

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
| GET | `/health` | Estado del servicio y si el modelo cargó y validó |
| GET | `/api/v1/modelo/info` | Métricas de NB07, fecha de entrenamiento, variables, limitaciones |
| GET | `/api/v1/distritos` | Los 103 distritos con código, nombre, `pct_urbano`, población |
| GET | `/api/v1/distritos/geojson` | GeoJSON para el mapa (ver sección 7 si falta) |
| POST | `/api/v1/predicciones` | Predicción para uno o varios distritos, rango de fechas y franjas |
| GET | `/api/v1/ranking` | Distritos ordenados por siniestros esperados en un rango, con su tercil |

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
`"observada"`).

En `/api/v1/ranking`, cada distrito trae su `tercil` (`alto`/`medio`/`bajo`)
y `confiabilidad_spearman`: el tercil medio ordena mal (~0.65) — es una
limitación documentada de NB07 §5.4, no un defecto de este endpoint. Esa
agrupación es **fija** (histórica, del conjunto de test 2025-2026 de NB07),
no se recalcula según el rango de fechas consultado — ver
`scripts/generar_terciles.py`.

---

## 6. El fallo silencioso de `log_tasa_base` (léase antes de tocar el modelo)

El modelo de frecuencia (`frecuencia_xgboost_offset_v2.pkl`) predice sobre un
**offset**: `base_margin = log_exposicion_v2 + log_tasa_base`.

`log_tasa_base = -12.936335060885538` **no está guardado dentro del
`.pkl`** — vive en `reports/results/nb07_especificacion.json`. Si se omite
al construir el `base_margin`, **XGBoost no lanza ningún error**: simplemente
predice ~415,000 veces más grande. Es el tipo de bug que pasa una revisión
manual porque el código "corre bien".

Por eso este servicio:

1. **Nunca hardcodea `log_tasa_base`** en el código de la aplicación — se
   lee de `nb07_especificacion.json` en cada arranque
   (`app/infrastructure/modelo/xgboost_adapter.py`).
2. Al arrancar, compara ese valor leído contra una constante de referencia y
   valida que el modelo tenga exactamente 8 features. Si algo no coincide,
   **el servicio se niega a arrancar** con un mensaje explícito — nunca
   arranca "degradado".
3. Al arrancar, corre una **predicción canaria** sobre una celda de
   referencia (San Salvador, 2024-06-15, franja Tarde) y verifica que caiga
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

**No existe ningún archivo GeoJSON de distritos en este repositorio.**
`GET /api/v1/distritos/geojson` devuelve **503** con un mensaje que indica
dónde colocarlo. Este servicio no inventa polígonos de ejemplo.

Para habilitarlo, colocar un GeoJSON válido (una `Feature` por distrito, con
la propiedad `distrito_key` o `codigo` para poder cruzarlo con
`GET /api/v1/distritos`) en:

```
data/geo/distritos.geojson
```

o apuntar `RUTA_GEOJSON_DISTRITOS` a otra ubicación.

---

## 8. Tests

```bash
pytest
```

54 tests, organizados así:

- `test_modelo_adapter.py` — arranque, validaciones, y la regresión del
  offset (sección 6).
- `test_exposicion_adapter.py`, `test_calendario.py`,
  `test_distrito_repository.py`, `test_tercil_repository.py`,
  `test_geojson_adapter.py` — cada adaptador de infraestructura contra los
  artefactos reales del repositorio.
- `test_casos_uso.py` — orquestación de `application/` con dobles de
  prueba en memoria (no toca el modelo real).
- `test_api.py` — end-to-end con `TestClient`, incluida la validación de
  entradas inválidas (distrito inexistente, fecha absurda, franja fuera de
  rango, rango de fechas demasiado largo).

Los terciles se regeneran (no corre en los tests ni en el arranque) con:

```bash
python scripts/generar_terciles.py
```

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
      modelo/          # XGBoost, exposición (NB05), calendario/feriados, clima, terciles
      geo/             # catálogo de distritos, GeoJSON
      web/             # routers FastAPI, schemas pydantic, dependencias
    config.py
    main.py
  scripts/
    generar_terciles.py  # precomputa terciles_nb07.json (ver sección 5)
  tests/
```

## 10. Limitaciones heredadas del modelo

Ver `GET /api/v1/modelo/info` para la lista completa (se sirve desde el
mismo lugar que documenta el código). Las más importantes:

- El tercil medio de distritos ordena mal (Spearman ~0.65).
- El dataset es sintético (NB01).
- La calibración tiene una desviación esperada del 2-3% (NB05 §6.5).
- Los feriados fuera de 2022-2026 se aproximan reciclando el patrón
  mes/día del último año conocido (exacto para feriados de fecha fija;
  aproximado para los móviles de Semana Santa).
- El submodelo de severidad (NB09) no se expone aquí: sus variables se
  conocen después de identificar a las víctimas y no admiten uso predictivo.
