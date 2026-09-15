"""Puertos del dominio (Protocol) — Python puro, sin dependencias de frameworks.

Cada adaptador de infrastructure/ implementa uno de estos protocolos. Los
casos de uso en application/ dependen solo de estas interfaces, nunca de las
implementaciones concretas.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from app.domain.models import Distrito, FeaturesCelda, Tercil


class PredictorFrecuenciaPort(Protocol):
    """Adaptador del modelo de frecuencia (XGBoost offset, NB07)."""

    def predecir_lote(self, filas: list[FeaturesCelda]) -> list[float]:
        """Siniestros esperados (lambda de Poisson) para cada fila, en el
        mismo orden. El adaptador ya validó al arrancar que el modelo tiene
        8 features y que log_tasa_base coincide con la especificación."""
        ...


class DistritoRepositoryPort(Protocol):
    def listar(self) -> list[Distrito]: ...

    def obtener(self, codigo: str) -> Distrito | None: ...

    def existe(self, codigo: str) -> bool: ...


class GeoDistritoRepositoryPort(Protocol):
    def obtener_geojson(self) -> dict | None:
        """None si el archivo de geometrías de distritos no existe en el
        repositorio — el router traduce eso a 503, nunca a un polígono
        inventado."""
        ...


class ClimaRepositoryPort(Protocol):
    def normal_climatologica(self, distrito_codigo: str, mes: int) -> float:
        """Promedio histórico de prcp_mensual para ese distrito y mes,
        sobre los años presentes en matriz_frecuencia_v2.csv (2022-2026)."""
        ...


class CalendarioRepositoryPort(Protocol):
    def es_feriado(self, fecha: date, distrito_codigo: str) -> bool:
        """Nacional (todo el país) o local San Salvador únicamente, según
        NB03 §5.2. Fuera de 2022-2026 repite el patrón mes/día conocido
        (asunción declarada en api/README.md: los feriados fijos no
        cambian de fecha; los móviles de Semana Santa se aproximan con la
        fecha observada más cercana del calendario conocido)."""
        ...


class ExposicionRepositoryPort(Protocol):
    def log_exposicion_v2(self, distrito_codigo: str, fecha: date) -> float:
        """log_exposicion (base NB03, constante por distrito) + ln(factor de
        crecimiento CAGR de NB05) evaluado en esa fecha."""
        ...


class TercilRepositoryPort(Protocol):
    def tercil_y_confiabilidad(self, distrito_codigo: str) -> tuple[Tercil, float]:
        """Agrupación fija de NB07 §5.4 (35/34/34 distritos por total
        observado en el conjunto de test 2025-2026), con el Spearman de
        ordenamiento de ese tercil como indicador de confiabilidad."""
        ...
