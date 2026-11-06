"""Adaptador de puntos individuales de siniestros (`siniestros.csv`, 89,946
filas reales — sintéticas, calibradas sobre agregados oficiales, ver NB01).

Para el mapa de hotspots que replica observatoriovial.fonat.gob.sv (clusters
de marcadores sobre mapa base real). Se carga en memoria una sola vez en el
lifespan; `buscar()` filtra en Python sobre esa lista (89,946 filas es rápido
de recorrer, no hace falta pandas ni un índice)."""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import date
from pathlib import Path

from app.domain.models import (
    FiltroSiniestros,
    Franja,
    OpcionesFiltroSiniestros,
    SiniestroPunto,
)


class SiniestroPuntoRepositoryError(RuntimeError):
    """`siniestros.csv` no existe o no tiene las columnas esperadas."""


_COLUMNAS_REQUERIDAS = {
    "id_siniestro", "fecha", "hora", "franja_horaria", "latitud", "longitud",
    "departamento", "distrito", "tipo_siniestro", "factor_causante",
    "tipo_vehiculo", "fallecidos", "lesionados",
}


# `siniestros.csv` escribe estas categorías sin tildes. Se muestran con tilde
# (opciones de filtro, filtros y puntos usan siempre el mismo texto). Mapa
# explícito: un valor que no esté aquí se deja tal cual, nunca se adivina.
_TIPO_SINIESTRO_CON_TILDES = {
    "Colision": "Colisión",
    "Caracteristicas especiales": "Características especiales",
}
_TIPO_VEHICULO_CON_TILDES = {
    "Automovil": "Automóvil",
    "Camion": "Camión",
}


def _fila_a_punto(
    fila: dict[str, str], nombre_distrito: Callable[[str, str], str] | None = None
) -> SiniestroPunto:
    distrito = fila["distrito"]
    if nombre_distrito is not None:
        distrito = nombre_distrito(distrito, fila["departamento"])
    return SiniestroPunto(
        id_siniestro=fila["id_siniestro"],
        fecha=date.fromisoformat(fila["fecha"]),
        hora=fila["hora"],
        franja=Franja(fila["franja_horaria"]),
        latitud=float(fila["latitud"]),
        longitud=float(fila["longitud"]),
        departamento=fila["departamento"],
        distrito=distrito,
        tipo_siniestro=_TIPO_SINIESTRO_CON_TILDES.get(fila["tipo_siniestro"], fila["tipo_siniestro"]),
        factor_causante=fila["factor_causante"],
        tipo_vehiculo=_TIPO_VEHICULO_CON_TILDES.get(fila["tipo_vehiculo"], fila["tipo_vehiculo"]),
        fallecidos=int(fila["fallecidos"]),
        lesionados=int(fila["lesionados"]),
    )


class SiniestroPuntoAdapter:
    def __init__(
        self,
        ruta_siniestros: Path,
        nombre_distrito: Callable[[str, str], str] | None = None,
    ) -> None:
        """`nombre_distrito(distrito, departamento)` devuelve el nombre para
        mostrar (con tildes). Opcional: sin él se usa el texto del CSV."""
        self._ruta = ruta_siniestros
        self._nombre_distrito = nombre_distrito
        self._puntos: list[SiniestroPunto] | None = None
        self._opciones: OpcionesFiltroSiniestros | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta.exists():
            raise SiniestroPuntoRepositoryError(f"No se encontró {self._ruta}")

        with self._ruta.open(encoding="utf-8-sig") as fh:
            lector = csv.DictReader(fh, delimiter=";")
            faltantes = _COLUMNAS_REQUERIDAS - set(lector.fieldnames or [])
            if faltantes:
                raise SiniestroPuntoRepositoryError(
                    f"{self._ruta} no tiene las columnas esperadas: faltan {faltantes}"
                )
            puntos = [_fila_a_punto(fila, self._nombre_distrito) for fila in lector]

        if not puntos:
            raise SiniestroPuntoRepositoryError(f"{self._ruta}: no tiene filas")

        self._puntos = puntos
        self._opciones = OpcionesFiltroSiniestros(
            tipos_siniestro=sorted({p.tipo_siniestro for p in puntos}),
            tipos_vehiculo=sorted({p.tipo_vehiculo for p in puntos}),
            factores_causantes=sorted({p.factor_causante for p in puntos}),
            departamentos=sorted({p.departamento for p in puntos}),
        )

    def _asegurar_cargado(self) -> list[SiniestroPunto]:
        if self._puntos is None:
            raise SiniestroPuntoRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar)."
            )
        return self._puntos

    def buscar(self, filtro: FiltroSiniestros) -> list[SiniestroPunto]:
        puntos = self._asegurar_cargado()

        def coincide(p: SiniestroPunto) -> bool:
            if filtro.fecha_inicio and p.fecha < filtro.fecha_inicio:
                return False
            if filtro.fecha_fin and p.fecha > filtro.fecha_fin:
                return False
            if filtro.tipo_siniestro and p.tipo_siniestro != filtro.tipo_siniestro:
                return False
            if filtro.tipo_vehiculo and p.tipo_vehiculo != filtro.tipo_vehiculo:
                return False
            if filtro.factor_causante and p.factor_causante != filtro.factor_causante:
                return False
            if filtro.departamento and p.departamento != filtro.departamento:
                return False
            if filtro.distrito and p.distrito != filtro.distrito:
                return False
            if filtro.hora_inicio and p.hora < filtro.hora_inicio:
                return False
            if filtro.hora_fin and p.hora > filtro.hora_fin:
                return False
            return True

        return [p for p in puntos if coincide(p)]

    def opciones_filtro(self) -> OpcionesFiltroSiniestros:
        self._asegurar_cargado()
        assert self._opciones is not None
        return self._opciones
