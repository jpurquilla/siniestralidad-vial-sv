"""Caso de uso: composición porcentual del tipo de vehículo en siniestros 2022-2024.

Se calcula de siniestros.csv (nunca a mano) una vez al arrancar. Solo alimenta los
supuestos/limitaciones del informe: el tipo de vehículo no varía de forma apreciable con la
franja, el distrito ni la causa, así que no se desglosa.

Python puro: lee los siniestros a través del puerto del dominio.
"""

from __future__ import annotations

from collections import Counter

from app.application.composicion_causas import (
    PERIODO_ENTRENAMIENTO_FIN,
    PERIODO_ENTRENAMIENTO_INICIO,
    ComposicionCausasError,
)
from app.domain.informe import ComposicionVehiculos, PorcentajeVehiculo
from app.domain.models import FiltroSiniestros
from app.domain.ports import SiniestroPuntoRepositoryPort


class CalcularComposicionVehiculos:
    def __init__(self, siniestros: SiniestroPuntoRepositoryPort) -> None:
        self._siniestros = siniestros

    def ejecutar(self) -> ComposicionVehiculos:
        filtro = FiltroSiniestros(
            fecha_inicio=PERIODO_ENTRENAMIENTO_INICIO, fecha_fin=PERIODO_ENTRENAMIENTO_FIN
        )
        conteos = Counter(p.tipo_vehiculo for p in self._siniestros.buscar(filtro))
        total = sum(conteos.values())
        if total == 0:
            raise ComposicionCausasError(
                f"No hay siniestros en {PERIODO_ENTRENAMIENTO_INICIO}..{PERIODO_ENTRENAMIENTO_FIN}."
            )
        ordenados = sorted(conteos.items(), key=lambda kv: (-kv[1], kv[0]))
        return ComposicionVehiculos(
            periodo_inicio=PERIODO_ENTRENAMIENTO_INICIO,
            periodo_fin=PERIODO_ENTRENAMIENTO_FIN,
            porcentajes=tuple(PorcentajeVehiculo(tipo, n / total * 100) for tipo, n in ordenados),
        )
