"""Caso de uso: agregar los siniestros (que cumplen un filtro) en hexágonos de
una grilla fija o por distrito.

Las ubicaciones del conjunto sintético no tienen precisión por debajo de ~1 km
(NB10 §8), así que estas agregaciones reemplazan a los puntos exactos. No
modifica ningún dato: solo cuenta.

Python puro: orquesta los puertos del dominio, sin importar FastAPI ni pandas.
"""

from __future__ import annotations

from app.domain import hexagonos
from app.domain.models import DistritoAgregado, FiltroSiniestros, HexagonoAgregado
from app.domain.ports import DistritoRepositoryPort, SiniestroPuntoRepositoryPort

TASA_POR = 10_000


class AgregarSiniestros:
    def __init__(
        self,
        siniestros: SiniestroPuntoRepositoryPort,
        distritos: DistritoRepositoryPort,
        lado_hexagono_km: float,
    ) -> None:
        self._siniestros = siniestros
        self._distritos = distritos
        self._lado_km = lado_hexagono_km

    @property
    def lado_hexagono_km(self) -> float:
        return self._lado_km

    def por_hexagono(self, filtro: FiltroSiniestros) -> list[HexagonoAgregado]:
        conteos: dict[hexagonos.Hexagono, int] = {}
        distritos: dict[hexagonos.Hexagono, set[str]] = {}
        for p in self._siniestros.buscar(filtro):
            h = hexagonos.hexagono_de(p.latitud, p.longitud, self._lado_km)
            conteos[h] = conteos.get(h, 0) + 1
            distritos.setdefault(h, set()).add(p.distrito)

        return [
            HexagonoAgregado(
                id=f"{q}_{r}",
                conteo=conteo,
                distritos=sorted(distritos[(q, r)]),
                centro=hexagonos.centro_de((q, r), self._lado_km),
                vertices=hexagonos.vertices_de((q, r), self._lado_km),
            )
            for (q, r), conteo in sorted(conteos.items())
        ]

    def por_distrito(self, filtro: FiltroSiniestros) -> list[DistritoAgregado]:
        """Los 103 distritos, también los que quedan en cero con ese filtro."""
        catalogo = self._distritos.listar()
        por_clave = {(d.nombre, d.departamento): d.codigo for d in catalogo}
        conteos = dict.fromkeys((d.codigo for d in catalogo), 0)
        for p in self._siniestros.buscar(filtro):
            conteos[por_clave[(p.distrito, p.departamento)]] += 1

        return [
            DistritoAgregado(
                distrito_codigo=d.codigo,
                nombre=d.nombre,
                departamento=d.departamento,
                siniestros=conteos[d.codigo],
                poblacion=d.poblacion,
                tasa_por_10mil=conteos[d.codigo] / d.poblacion * TASA_POR if d.poblacion else 0.0,
            )
            for d in catalogo
        ]
