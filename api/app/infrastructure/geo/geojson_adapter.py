"""Adaptador de geometrías de distritos.

No hay ningún GeoJSON de distritos versionado en el repositorio (ver
api/README.md, sección GeoJSON). Este adaptador NO inventa polígonos de
ejemplo: si el archivo no existe, devuelve None y el router traduce eso a
503 con un mensaje que dice exactamente qué archivo falta y dónde ubicarlo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GeoJsonAdapter:
    """Implementa GeoDistritoRepositoryPort."""

    def __init__(self, ruta_geojson: Path) -> None:
        self._ruta_geojson = ruta_geojson

    def obtener_geojson(self) -> dict | None:
        if not self._ruta_geojson.exists():
            logger.warning(
                "GeoJSON de distritos no encontrado en %s — /distritos/geojson devolverá 503.",
                self._ruta_geojson,
            )
            return None
        return json.loads(self._ruta_geojson.read_text(encoding="utf-8"))
