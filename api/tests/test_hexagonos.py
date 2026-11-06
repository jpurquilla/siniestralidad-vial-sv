"""Grilla hexagonal (dominio puro)."""

from __future__ import annotations

import math

import pytest

from app.domain import hexagonos

LADO_KM = 2.0


def _distancia_km(a, b) -> float:
    x1, y1 = hexagonos.a_km(*a)
    x2, y2 = hexagonos.a_km(*b)
    return math.hypot(x2 - x1, y2 - y1)


class TestHexagonos:
    def test_ida_y_vuelta_de_la_proyeccion(self):
        x, y = hexagonos.a_km(13.7, -89.2)
        lat, lon = hexagonos.a_lat_lon(x, y)
        assert (lat, lon) == pytest.approx((13.7, -89.2), abs=1e-9)

    def test_el_centro_pertenece_a_su_propio_hexagono(self):
        for h in [(0, 0), (5, -3), (-12, 9)]:
            lat, lon = hexagonos.centro_de(h, LADO_KM)
            assert hexagonos.hexagono_de(lat, lon, LADO_KM) == h

    def test_el_lado_mide_2_km(self):
        v = hexagonos.vertices_de((3, 4), LADO_KM)
        assert len(v) == 6
        for i in range(6):
            assert _distancia_km(v[i], v[(i + 1) % 6]) == pytest.approx(LADO_KM, rel=1e-6)

    def test_el_centro_a_cada_vertice_mide_el_lado(self):
        c = hexagonos.centro_de((3, 4), LADO_KM)
        for v in hexagonos.vertices_de((3, 4), LADO_KM):
            assert _distancia_km(c, v) == pytest.approx(LADO_KM, rel=1e-6)

    def test_hexagonos_vecinos_comparten_arista_sin_solaparse(self):
        # Centros de hexágonos adyacentes distan sqrt(3) × lado.
        a = hexagonos.centro_de((0, 0), LADO_KM)
        for vecino in [(1, 0), (0, 1), (-1, 1)]:
            assert _distancia_km(a, hexagonos.centro_de(vecino, LADO_KM)) == pytest.approx(
                math.sqrt(3) * LADO_KM, rel=1e-6
            )

    def test_puntos_muy_cercanos_caen_en_el_mismo_hexagono(self):
        lat, lon = hexagonos.centro_de((2, 2), LADO_KM)
        assert hexagonos.hexagono_de(lat + 0.0005, lon - 0.0005, LADO_KM) == (2, 2)
