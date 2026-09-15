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

    # --- Artefactos del modelo (NB07) ---
    ruta_modelo_frecuencia: Path = _RAIZ_REPO / "models" / "frecuencia_xgboost_offset_v2.pkl"
    ruta_especificacion_nb07: Path = _RAIZ_REPO / "reports" / "results" / "nb07_especificacion.json"
    ruta_metricas_nb07: Path = _RAIZ_REPO / "reports" / "results" / "nb07_metricas.json"

    # --- Tendencia de exposición (NB05) ---
    ruta_especificacion_nb05: Path = _RAIZ_REPO / "reports" / "results" / "nb05_especificacion_v2.json"

    # --- Datos base para el catálogo de distritos, clima y terciles ---
    ruta_matriz_frecuencia: Path = _RAIZ_REPO / "data" / "processed" / "matriz_frecuencia_v2.csv"
    ruta_feriados: Path = _RAIZ_REPO / "data" / "raw" / "feriados_elsalvador_2022_2026.xlsx"
    ruta_siniestros: Path = _RAIZ_REPO / "data" / "raw" / "siniestros.csv"

    # --- GeoJSON de distritos: puede no existir (ver README, sección GeoJSON) ---
    ruta_geojson_distritos: Path = _RAIZ_REPO / "data" / "geo" / "distritos.geojson"

    # --- Terciles precomputados (ver infrastructure/modelo/terciles_nb07.json) ---
    ruta_terciles: Path = (
        Path(__file__).resolve().parent / "infrastructure" / "modelo" / "terciles_nb07.json"
    )

    # --- Validación de rango de fechas (NB03 §5.2: feriados 2022-2026 conocidos;
    # fuera de ese rango se reciclan por mes/día, ver CalendarioAdapter) ---
    fecha_minima: str = "2022-01-01"
    fecha_maxima: str = "2100-12-31"
    max_dias_por_rango: int = 366

    # --- CORS (para el frontend del prototipo) ---
    cors_origenes: list[str] = ["*"]

    # --- Logging ---
    nivel_log: str = "INFO"

    # --- Metadatos de la app ---
    nombre_app: str = "API de predicción de siniestralidad vial — El Salvador"
    version_app: str = "0.1.0"


settings = Settings()
