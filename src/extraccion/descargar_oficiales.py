"""Descarga los archivos oficiales del Censo 2024 (BCR/ONEC) que usa la etapa final.

  TAB_HOG_4.xlsx                 -> cfg.CENSO_BIENES_FILE   (hogares con automóvil, bicicleta, motocicleta)
  TAB_CE_5.xlsx                  -> cfg.CENSO_TRABAJO_FILE  (ocupados según lugar de trabajo)
  distritos_censo2024.geojson    -> cfg.DISTRITOS_GEO_FILE  (capa «Población total por distrito», 262 polígonos)
Los archivos se guardan tal como los publica el BCR, sin modificaciones.
Índice de tabulados: https://poblacion.bcr.gob.sv/pages/teg-base-de-datos-y-tabulados

Uso: python -m src.extraccion.descargar_oficiales
"""
import json
from src import config as cfg
from ._web import get

BASE = "https://censo2024.bcr.gob.sv/wp-content/uploads/tablas-geoportal/2025"
CAPA = ("https://services8.arcgis.com/xSn1Pefr3M9zkw7Z/arcgis/rest/services/"
        "capas_poblaci%C3%B3n_1/FeatureServer/6/query")


def main():
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True); cfg.GEO_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CENSO_BIENES_FILE.write_bytes(get(f"{BASE}/TAB_HOG_4.xlsx", binario=True))
    cfg.CENSO_TRABAJO_FILE.write_bytes(get(f"{BASE}/TAB_CE_5.xlsx", binario=True))
    gj = get(CAPA, params={"where": "1=1", "outFields": "*", "returnGeometry": "true",
                           "outSR": 4326, "geometryPrecision": 6, "f": "geojson"})
    assert len(json.loads(gj)["features"]) == 262
    cfg.DISTRITOS_GEO_FILE.write_text(gj, encoding="utf-8")
    print("OK:", cfg.CENSO_BIENES_FILE.name, cfg.CENSO_TRABAJO_FILE.name, cfg.DISTRITOS_GEO_FILE.name)


if __name__ == "__main__":
    main()
