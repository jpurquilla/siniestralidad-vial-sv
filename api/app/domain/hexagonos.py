"""Grilla hexagonal para agregar siniestros — Python puro, sin dependencias.

Los hexágonos son de "punta arriba" (pointy-top) y su LADO mide `lado_km`
(no el ancho: un hexágono de 2 km de lado mide ~3.46 km de lado a lado).

Proyección: equirrectangular local alrededor de El Salvador (lat0 = 13.75°,
lon0 = -88.9°). En un país de ~1.3° × 2.4° la deformación es menor que 1%
(el coseno de la latitud varía de 0.974 a 0.968), de modo que el "2 km" es
exacto en el centro y aproximado en ±0.6% en los bordes. Es una agregación
visual: no hace falta una proyección UTM ni una librería geoespacial.
"""

from __future__ import annotations

import math

RADIO_TIERRA_KM = 6371.0088
LAT_REFERENCIA = 13.75
LON_REFERENCIA = -88.9

_COS_LAT = math.cos(math.radians(LAT_REFERENCIA))
_RAIZ3 = math.sqrt(3.0)

Hexagono = tuple[int, int]  # coordenadas axiales (q, r)


def a_km(latitud: float, longitud: float) -> tuple[float, float]:
    """(x, y) en km respecto del punto de referencia (x al este, y al norte)."""
    x = math.radians(longitud - LON_REFERENCIA) * RADIO_TIERRA_KM * _COS_LAT
    y = math.radians(latitud - LAT_REFERENCIA) * RADIO_TIERRA_KM
    return x, y


def a_lat_lon(x_km: float, y_km: float) -> tuple[float, float]:
    latitud = LAT_REFERENCIA + math.degrees(y_km / RADIO_TIERRA_KM)
    longitud = LON_REFERENCIA + math.degrees(x_km / (RADIO_TIERRA_KM * _COS_LAT))
    return latitud, longitud


def _redondear_axial(q: float, r: float) -> Hexagono:
    """Redondeo de coordenadas cúbicas fraccionarias al hexágono más cercano."""
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def hexagono_de(latitud: float, longitud: float, lado_km: float) -> Hexagono:
    """Hexágono (q, r) que contiene el punto."""
    x, y = a_km(latitud, longitud)
    q = (_RAIZ3 / 3.0 * x - y / 3.0) / lado_km
    r = (2.0 / 3.0 * y) / lado_km
    return _redondear_axial(q, r)


def centro_de(hexagono: Hexagono, lado_km: float) -> tuple[float, float]:
    """(latitud, longitud) del centro del hexágono."""
    q, r = hexagono
    x = lado_km * _RAIZ3 * (q + r / 2.0)
    y = lado_km * 1.5 * r
    return a_lat_lon(x, y)


def vertices_de(hexagono: Hexagono, lado_km: float) -> list[tuple[float, float]]:
    """Los 6 vértices (latitud, longitud), en sentido antihorario."""
    q, r = hexagono
    cx = lado_km * _RAIZ3 * (q + r / 2.0)
    cy = lado_km * 1.5 * r
    vertices = []
    for i in range(6):
        angulo = math.radians(60 * i - 30)
        vertices.append(a_lat_lon(cx + lado_km * math.cos(angulo), cy + lado_km * math.sin(angulo)))
    return vertices
