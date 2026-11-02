"""Puertos del dominio (Protocol) — Python puro, sin dependencias de frameworks.

Cada adaptador de infrastructure/ implementa uno de estos protocolos. Los
casos de uso en application/ dependen solo de estas interfaces, nunca de las
implementaciones concretas.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from app.domain.models import (
    ComparacionDistrito,
    Distrito,
    FeaturesCelda,
    FiltroSiniestros,
    ImpactoAnual,
    ImpactoResumenItem,
    LetalidadPorEdad,
    OpcionesFiltroSiniestros,
    RepartoTipoUsuario,
    SiniestroPunto,
    Tercil,
)


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


class ImpactoLeyRepositoryPort(Protocol):
    """Datos reales de NB06 (impacto de la Ley Cero Tolerancia, vigente
    desde 2024-12-20). No hay nada que predecir acá: es la serie oficial."""

    def serie_anual(self) -> list[ImpactoAnual]: ...

    def fecha_aprobacion_ley(self) -> date: ...

    def resumen(self) -> list[ImpactoResumenItem]: ...

    def limitaciones(self) -> list[str]: ...


class UsuariosVulnerablesRepositoryPort(Protocol):
    """Retrospectivo (NB09 §4): agregados de matriz_severidad.csv. Nunca
    predictivo — `grupo_vulnerable` y `rango_etario` se conocen después de
    identificar a las víctimas."""

    def reparto_por_tipo(self) -> list[RepartoTipoUsuario]: ...

    def letalidad_por_edad(self) -> list[LetalidadPorEdad]: ...


class ComparacionHistoricaRepositoryPort(Protocol):
    def por_distrito(self) -> list[ComparacionDistrito]:
        """Observado (matriz_frecuencia_v2.csv) contra predicho por el
        modelo de NB07, sumados por distrito sobre todo el histórico."""
        ...


class SiniestroPuntoRepositoryPort(Protocol):
    """Puntos individuales reales (`siniestros.csv`, 89,946 filas) para el
    mapa de hotspots — replica observatoriovial.fonat.gob.sv. NB10 §8: sin
    estructura espacial real bajo ~1 km; los clusters no son un hallazgo de
    concentración vial, se declara así en la propia pantalla."""

    def buscar(self, filtro: FiltroSiniestros) -> list[SiniestroPunto]: ...

    def opciones_filtro(self) -> OpcionesFiltroSiniestros:
        """Valores reales distintos para poblar los desplegables de
        filtro — nunca hardcodeados en el frontend."""
        ...
