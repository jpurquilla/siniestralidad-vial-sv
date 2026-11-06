"""Caso de uso: panel de equidad — calibración (sum(predicho)/sum(observado);
; >1 = el modelo sobreestima)
y Spearman de orden predicho-vs-observado dentro de dos cortes ya
establecidos en el proyecto (nunca inventados acá):

- Urbano/rural, por el mismo umbral de `pct_urbano` que NB08 §4
  (0.3 — ver `settings.umbral_pct_urbano_equidad`).
- Alta/baja densidad poblacional, por el flag `poblacion_baja` que el
  propio modelo final (NB12) ya usa como variable.

Python puro: usa `statistics.correlation(..., method="ranked")` (stdlib,
Python 3.12+) para Spearman — no hace falta scipy."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from app.domain.models import ComparacionDistrito, Distrito, GrupoEquidad
from app.domain.ports import ComparacionHistoricaRepositoryPort, DistritoRepositoryPort


@dataclass(frozen=True)
class ResultadoEquidad:
    urbano_rural: list[GrupoEquidad]
    densidad_poblacional: list[GrupoEquidad]


def _calcular_grupo(nombre: str, comparaciones: list[ComparacionDistrito]) -> GrupoEquidad:
    if not comparaciones:
        return GrupoEquidad(grupo=nombre, calibracion=0.0, spearman=0.0, n_distritos=0)

    total_observado = sum(c.observado for c in comparaciones)
    total_predicho = sum(c.predicho for c in comparaciones)
    calibracion = total_predicho / total_observado if total_observado else 0.0

    if len(comparaciones) > 1:
        observados = [c.observado for c in comparaciones]
        predichos = [c.predicho for c in comparaciones]
        spearman = statistics.correlation(observados, predichos, method="ranked")
    else:
        spearman = 0.0

    return GrupoEquidad(
        grupo=nombre, calibracion=calibracion, spearman=spearman, n_distritos=len(comparaciones)
    )


class CalcularEquidad:
    def __init__(
        self,
        comparacion_historica: ComparacionHistoricaRepositoryPort,
        distritos: DistritoRepositoryPort,
        umbral_pct_urbano: float,
    ) -> None:
        self._comparacion = comparacion_historica
        self._distritos = distritos
        self._umbral = umbral_pct_urbano

    def ejecutar(self) -> ResultadoEquidad:
        comp_por_codigo = {c.distrito_codigo: c for c in self._comparacion.por_distrito()}

        def comparaciones_de(distritos_del_grupo: list[Distrito]) -> list[ComparacionDistrito]:
            return [comp_por_codigo[d.codigo] for d in distritos_del_grupo if d.codigo in comp_por_codigo]

        todos = self._distritos.listar()
        urbanos = [d for d in todos if d.pct_urbano >= self._umbral]
        rurales = [d for d in todos if d.pct_urbano < self._umbral]
        alta_densidad = [d for d in todos if not d.poblacion_baja]
        baja_densidad = [d for d in todos if d.poblacion_baja]

        return ResultadoEquidad(
            urbano_rural=[
                _calcular_grupo("Urbano", comparaciones_de(urbanos)),
                _calcular_grupo("Rural", comparaciones_de(rurales)),
            ],
            densidad_poblacional=[
                _calcular_grupo("Alta densidad", comparaciones_de(alta_densidad)),
                _calcular_grupo("Baja densidad", comparaciones_de(baja_densidad)),
            ],
        )
