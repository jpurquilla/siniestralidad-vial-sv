"""Composición de causas por franja: proporciones históricas 2022-2024."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from app.application.composicion_causas import CalcularComposicionCausas, ComposicionCausasError
from app.domain.models import Franja
from app.infrastructure.analitica.siniestro_punto_adapter import SiniestroPuntoAdapter

_COLUMNAS = [
    "id_siniestro", "anio", "mes", "dia", "fecha", "dia_semana", "hora", "franja_horaria",
    "latitud", "longitud", "departamento", "distrito", "municipio_2024", "zona", "tipo_via",
    "tipo_siniestro", "factor_causante", "condicion_via", "tipo_vehiculo", "fallecidos",
    "lesionados", "registro_parcial",
]

_HORA = {
    Franja.MADRUGADA: "03:00",
    Franja.MANANA: "08:00",
    Franja.TARDE: "14:00",
    Franja.NOCHE: "20:00",
}


def _fila(i: int, fecha: str, franja: Franja, causa: str) -> dict[str, str]:
    return {
        "id_siniestro": f"SV-{i}", "anio": fecha[:4], "mes": fecha[5:7], "dia": fecha[8:10],
        "fecha": fecha, "dia_semana": "Lunes", "hora": _HORA[franja], "franja_horaria": franja.value,
        "latitud": "13.7", "longitud": "-89.2", "departamento": "San Salvador",
        "distrito": "San Salvador", "municipio_2024": "San Salvador Centro", "zona": "Urbana",
        "tipo_via": "Calle", "tipo_siniestro": "Colision", "factor_causante": causa,
        "condicion_via": "Seca", "tipo_vehiculo": "Automovil", "fallecidos": "0", "lesionados": "0",
        "registro_parcial": "No",
    }


def _adaptador(tmp_path: Path, filas: list[dict[str, str]]) -> SiniestroPuntoAdapter:
    ruta = tmp_path / "siniestros.csv"
    with ruta.open("w", encoding="utf-8-sig", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=_COLUMNAS, delimiter=";")
        escritor.writeheader()
        escritor.writerows(filas)
    adaptador = SiniestroPuntoAdapter(ruta)
    adaptador.cargar_y_validar()
    return adaptador


def _filas_base() -> list[dict[str, str]]:
    """Por franja: 6 siniestros de 2023 (4 de «Velocidad Excesiva», 2 de «Distracción del Conductor»)."""
    filas, i = [], 0
    for franja in Franja:
        for causa, n in (("Velocidad Excesiva", 4), ("Distracción del Conductor", 2)):
            for _ in range(n):
                i += 1
                filas.append(_fila(i, "2023-05-10", franja, causa))
    return filas


class TestCalcularComposicionCausas:
    def test_las_proporciones_de_cada_franja_suman_1_y_van_de_mayor_a_menor(self, tmp_path):
        composicion = CalcularComposicionCausas(_adaptador(tmp_path, _filas_base())).ejecutar()
        for franja in Franja:
            proporciones = composicion.proporciones(franja)
            assert sum(p.proporcion for p in proporciones) == pytest.approx(1.0)
            assert [p.proporcion for p in proporciones] == sorted((p.proporcion for p in proporciones), reverse=True)
            assert proporciones[0].causa == "Velocidad Excesiva"
            assert proporciones[0].proporcion == pytest.approx(4 / 6)

    def test_solo_cuenta_2022_2024_agregar_2025_y_2026_no_cambia_las_proporciones(self, tmp_path):
        (tmp_path / "a").mkdir()
        base = CalcularComposicionCausas(_adaptador(tmp_path / "a", _filas_base())).ejecutar()

        filas = _filas_base()
        for franja in Franja:  # muchos siniestros posteriores de otra causa que cambiarían todo si contaran
            for k in range(50):
                filas.append(_fila(1000 + k, "2025-03-01", franja, "Estado de Ebriedad o Droga"))
                filas.append(_fila(2000 + k, "2026-01-15", franja, "Estado de Ebriedad o Droga"))
        (tmp_path / "b").mkdir()
        con_2025 = CalcularComposicionCausas(_adaptador(tmp_path / "b", filas)).ejecutar()

        assert con_2025.por_franja == base.por_franja
        assert con_2025.n_siniestros == base.n_siniestros
        assert all(p.causa != "Estado de Ebriedad o Droga" for f in Franja for p in con_2025.proporciones(f))

    def test_incluye_los_limites_del_periodo(self, tmp_path):
        filas = _filas_base()
        for franja in Franja:
            filas.append(_fila(900, "2022-01-01", franja, "Falla Mecánica"))
            filas.append(_fila(901, "2024-12-31", franja, "Falla Mecánica"))
        composicion = CalcularComposicionCausas(_adaptador(tmp_path, filas)).ejecutar()
        assert composicion.periodo_inicio == date(2022, 1, 1) and composicion.periodo_fin == date(2024, 12, 31)
        assert composicion.n_siniestros[Franja.NOCHE] == 8

    def test_una_franja_sin_siniestros_es_un_error_claro(self, tmp_path):
        filas = [f for f in _filas_base() if f["franja_horaria"] != Franja.NOCHE.value]
        with pytest.raises(ComposicionCausasError, match="Noche"):
            CalcularComposicionCausas(_adaptador(tmp_path, filas)).ejecutar()

    def test_descomponer_reparte_exactamente_la_prediccion(self, tmp_path):
        composicion = CalcularComposicionCausas(_adaptador(tmp_path, _filas_base())).ejecutar()
        por_causa = composicion.descomponer(0.72, Franja.NOCHE)
        assert sum(c.esperado for c in por_causa) == pytest.approx(0.72)
        assert por_causa[0].esperado == pytest.approx(0.72 * 4 / 6)
