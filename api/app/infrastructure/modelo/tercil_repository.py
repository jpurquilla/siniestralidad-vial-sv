"""Adaptador de terciles: lee el agrupamiento fijo precomputado por
api/scripts/generar_terciles.py (ver ese script para el protocolo, que
replica NB07 §5.4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.models import Tercil

logger = logging.getLogger(__name__)


class TercilRepositoryError(RuntimeError):
    """terciles_nb07.json no existe o no tiene la forma esperada."""


class TercilRepository:
    """Implementa TercilRepositoryPort."""

    def __init__(self, ruta_terciles: Path) -> None:
        self._ruta_terciles = ruta_terciles
        self._tercil_por_distrito: dict[str, Tercil] | None = None
        self._confiabilidad_por_tercil: dict[Tercil, float] | None = None

    def cargar_y_validar(self) -> None:
        if not self._ruta_terciles.exists():
            raise TercilRepositoryError(
                f"No se encontró {self._ruta_terciles}. Generarlo con "
                "api/scripts/generar_terciles.py."
            )

        try:
            data = json.loads(self._ruta_terciles.read_text(encoding="utf-8"))
            tercil_por_distrito = {
                distrito: Tercil(valor) for distrito, valor in data["tercil_por_distrito"].items()
            }
            confiabilidad_por_tercil = {
                Tercil(tercil): float(info["spearman"])
                for tercil, info in data["confiabilidad_por_tercil"].items()
            }
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise TercilRepositoryError(
                f"{self._ruta_terciles.name} no tiene la forma esperada: {exc}"
            ) from exc

        if len(tercil_por_distrito) != 103:
            raise TercilRepositoryError(
                f"Se esperaban 103 distritos en {self._ruta_terciles.name}, "
                f"hay {len(tercil_por_distrito)}."
            )
        faltantes = set(Tercil) - set(confiabilidad_por_tercil)
        if faltantes:
            raise TercilRepositoryError(
                f"Falta confiabilidad para los terciles {faltantes} en {self._ruta_terciles.name}"
            )

        self._tercil_por_distrito = tercil_por_distrito
        self._confiabilidad_por_tercil = confiabilidad_por_tercil

        logger.info(
            "Terciles NB07 validados: %d distritos, confiabilidad=%s",
            len(tercil_por_distrito),
            {t.value: v for t, v in confiabilidad_por_tercil.items()},
        )

    def tercil_y_confiabilidad(self, distrito_codigo: str) -> tuple[Tercil, float]:
        if self._tercil_por_distrito is None or self._confiabilidad_por_tercil is None:
            raise TercilRepositoryError(
                "El adaptador no fue validado (llamar cargar_y_validar) antes de usarse."
            )
        try:
            tercil = self._tercil_por_distrito[distrito_codigo]
        except KeyError as exc:
            raise TercilRepositoryError(
                f"Distrito desconocido en {self._ruta_terciles.name}: {distrito_codigo!r}"
            ) from exc
        return tercil, self._confiabilidad_por_tercil[tercil]
