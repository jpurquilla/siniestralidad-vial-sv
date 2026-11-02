"""Adaptador de usuarios vulnerables (NB09 §4, retrospectivo — nunca
predictivo: `grupo_vulnerable` y `rango_etario` se conocen después de
identificar a las víctimas).

Agrega `matriz_severidad.csv` (89,946 siniestros reales del conjunto
sintético). Metodología verificada contra los números ya publicados en
NB09_RESUMEN.md antes de escribir este adaptador (ver el docstring de
tests/test_usuarios_vulnerables_adapter.py para el detalle): la letalidad de
un grupo es la fracción de sus siniestros que tuvieron al menos un
fallecido — no el promedio de fallecidos por siniestro."""

from __future__ import annotations

import csv
from pathlib import Path

from app.domain.models import LetalidadPorEdad, RepartoTipoUsuario

_GRUPO_SIN_VICTIMAS = "Sin victimas"

# Nombre para mostrar (con tilde) de los grupos tal como vienen en la fuente.
# La clave interna de la fuente no cambia; solo lo que se expone.
_NOMBRE_PARA_MOSTRAR = {"Peaton": "Peatón"}

_ORDEN_RANGOS_ETARIOS = [
    "0-15",
    "16-20",
    "21-25",
    "26-30",
    "31-35",
    "36-40",
    "41-45",
    "46-50",
    "51-55",
    "56-60",
    "61-65",
    "66-70",
    "71+",
]


class UsuariosVulnerablesRepositoryError(RuntimeError):
    """`matriz_severidad.csv` no existe o no tiene las columnas esperadas."""


class UsuariosVulnerablesAdapter:
    def __init__(self, ruta_matriz_severidad: Path) -> None:
        self._ruta = ruta_matriz_severidad
        self._reparto_por_tipo: list[RepartoTipoUsuario] | None = None
        self._letalidad_por_edad: list[LetalidadPorEdad] | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta.exists():
            raise UsuariosVulnerablesRepositoryError(f"No se encontró {self._ruta}")

        conteo_por_tipo: dict[str, dict[str, int]] = {}
        conteo_por_edad: dict[str, dict[str, int]] = {}

        with self._ruta.open(encoding="utf-8") as fh:
            lector = csv.DictReader(fh)
            columnas_faltantes = {"grupo_vulnerable", "rango_etario", "fallecidos"} - set(
                lector.fieldnames or []
            )
            if columnas_faltantes:
                raise UsuariosVulnerablesRepositoryError(
                    f"{self._ruta} no tiene las columnas esperadas: faltan {columnas_faltantes}"
                )

            for fila in lector:
                grupo = fila["grupo_vulnerable"]
                if grupo == _GRUPO_SIN_VICTIMAS:
                    continue
                fue_fatal = int(fila["fallecidos"]) > 0

                bucket_tipo = conteo_por_tipo.setdefault(grupo, {"n": 0, "con_fallecido": 0})
                bucket_tipo["n"] += 1
                bucket_tipo["con_fallecido"] += int(fue_fatal)

                rango = fila["rango_etario"]
                bucket_edad = conteo_por_edad.setdefault(rango, {"n": 0, "con_fallecido": 0})
                bucket_edad["n"] += 1
                bucket_edad["con_fallecido"] += int(fue_fatal)

        if not conteo_por_tipo:
            raise UsuariosVulnerablesRepositoryError(f"{self._ruta}: no hay siniestros con víctimas")

        total_con_fallecido = sum(b["con_fallecido"] for b in conteo_por_tipo.values())

        self._reparto_por_tipo = [
            RepartoTipoUsuario(
                tipo=_NOMBRE_PARA_MOSTRAR.get(tipo, tipo),
                pct_fatales=b["con_fallecido"] / total_con_fallecido * 100,
                letalidad_por_siniestro=b["con_fallecido"] / b["n"],
            )
            for tipo, b in conteo_por_tipo.items()
        ]
        self._reparto_por_tipo.sort(key=lambda r: r.pct_fatales, reverse=True)

        self._letalidad_por_edad = [
            LetalidadPorEdad(
                rango_etario=rango,
                victimas=conteo_por_edad[rango]["n"],
                letalidad=conteo_por_edad[rango]["con_fallecido"] / conteo_por_edad[rango]["n"],
            )
            for rango in _ORDEN_RANGOS_ETARIOS
            if rango in conteo_por_edad
        ]

    def _asegurar_cargado(self) -> None:
        if self._reparto_por_tipo is None:
            raise UsuariosVulnerablesRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar)."
            )

    def reparto_por_tipo(self) -> list[RepartoTipoUsuario]:
        self._asegurar_cargado()
        assert self._reparto_por_tipo is not None
        return list(self._reparto_por_tipo)

    def letalidad_por_edad(self) -> list[LetalidadPorEdad]:
        self._asegurar_cargado()
        assert self._letalidad_por_edad is not None
        return list(self._letalidad_por_edad)
