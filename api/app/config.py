"""Configuración de la API — pydantic-settings, todo por variables de entorno.

Ningún secreto ni ruta absoluta va hardcodeada aquí: todo tiene un default
razonable relativo a la raíz del repositorio, sobreescribible por env var o
por un archivo .env (ver .env.example).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# api/app/config.py -> parents: app/, api/, <raíz del repo>
_RAIZ_REPO = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Modelo final (NB12): el archivo .pkl y sus columnas se resuelven
    # desde la especificación (campo modelo_final), no se fijan aquí ---
    directorio_modelos: Path = _RAIZ_REPO / "models"
    ruta_especificacion_nb12: Path = _RAIZ_REPO / "reports" / "results" / "nb12_especificacion.json"
    ruta_metricas_nb12: Path = _RAIZ_REPO / "reports" / "results" / "nb12_metricas.json"

    # --- Agregación de siniestros en hexágonos (mapa de concentraciones por área) ---
    lado_hexagono_km: float = 2.0

    # --- Tendencia de exposición (NB05) ---
    ruta_especificacion_nb05: Path = _RAIZ_REPO / "reports" / "results" / "nb05_especificacion_v2.json"

    # --- Datos base para el catálogo de distritos, clima y terciles ---
    ruta_matriz_frecuencia: Path = _RAIZ_REPO / "data" / "processed" / "matriz_frecuencia_v3.csv.gz"
    ruta_censo_poblacion: Path = _RAIZ_REPO / "data" / "raw" / "TAB_POB_AREA_1.xlsx"
    ruta_parque_vehicular: Path = _RAIZ_REPO / "data" / "raw" / "parque_vehicular_onasevi.csv"
    ruta_feriados: Path = _RAIZ_REPO / "data" / "raw" / "feriados_elsalvador_2022_2026.xlsx"
    ruta_siniestros: Path = _RAIZ_REPO / "data" / "raw" / "siniestros.csv"

    # --- GeoJSON de distritos: puede no existir (ver README, sección GeoJSON) ---
    ruta_geojson_distritos: Path = _RAIZ_REPO / "data" / "geo" / "distritos.geojson"
    # GeoJSON del censo 2024 (262 distritos, propiedades Dep_min/Dis_min): el mapa de la portada del PDF.
    ruta_geojson_censo: Path = _RAIZ_REPO / "data" / "geo" / "distritos_censo2024.geojson"

    # --- Análisis retrospectivos (NB06, NB09) — nunca predictivos ---
    ruta_resultados_nb06: Path = _RAIZ_REPO / "reports" / "results" / "nb06_resultados.json"
    ruta_matriz_severidad: Path = _RAIZ_REPO / "data" / "processed" / "matriz_severidad.csv"

    # --- Umbral urbano/rural para el panel de equidad (mismo corte que NB08 §4) ---
    umbral_pct_urbano_equidad: float = 0.3

    # --- Terciles precomputados con el modelo final (ver scripts/generar_terciles.py) ---
    ruta_terciles: Path = (
        Path(__file__).resolve().parent / "infrastructure" / "modelo" / "terciles_nb12.json"
    )

    # --- Validación de rango de fechas (NB03 §5.2: feriados 2022-2026 conocidos;
    # fuera de ese rango se reciclan por mes/día, ver CalendarioAdapter) ---
    fecha_minima: str = "2022-01-01"
    fecha_maxima: str = "2100-12-31"
    max_dias_por_rango: int = 366

    # --- CORS (para el frontend del prototipo). Sin credenciales: la API no usa
    # cookies ni sesiones, así que no hace falta allow_credentials. En env se
    # define como lista JSON: CORS_ORIGENES=["http://localhost:4200"] ---
    cors_origenes: list[str] = ["http://localhost:4200"]

    # --- Zona horaria del informe por periodo (campo `generado`) ---
    zona_horaria: str = "America/El_Salvador"

    # --- Logging ---
    nivel_log: str = "INFO"

    # --- Metadatos de la app ---
    nombre_app: str = "API de predicción de siniestralidad vial — El Salvador"
    version_app: str = "0.1.0"


settings = Settings()
