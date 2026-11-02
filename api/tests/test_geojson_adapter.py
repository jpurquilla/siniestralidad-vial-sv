"""El GeoJSON de distritos no existe en este repositorio (ver README) — el
adaptador debe devolver None, nunca inventar polígonos."""

from __future__ import annotations

import json

from app.infrastructure.geo.geojson_adapter import GeoJsonAdapter


def test_devuelve_none_si_no_existe(tmp_path):
    adapter = GeoJsonAdapter(tmp_path / "no_existe.geojson")
    assert adapter.obtener_geojson() is None


def test_devuelve_el_contenido_si_existe(tmp_path):
    ruta = tmp_path / "distritos.geojson"
    contenido = {"type": "FeatureCollection", "features": []}
    ruta.write_text(json.dumps(contenido), encoding="utf-8")

    adapter = GeoJsonAdapter(ruta)
    assert adapter.obtener_geojson() == contenido
