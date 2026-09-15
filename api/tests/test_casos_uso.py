"""Tests de los casos de uso (application/) con dobles de prueba en memoria
para los puertos — no tocan el modelo real ni ningún archivo. Verifican
orquestación pura: qué features se construyen y cómo se agregan los
resultados, no la corrección numérica del modelo (eso ya lo cubren los
tests de infrastructure/)."""

from __future__ import annotations

from datetime import date

import pytest

from app.application.predecir_celda import (
    DistritoNoEncontradoError,
    PeticionPrediccionCelda,
    PredecirCelda,
)
from app.application.ranking_distritos import PeticionRanking, RankingDistritos
from app.domain.models import Distrito, FeaturesCelda, Franja, FuentePrecipitacion, Tercil


class PredictorFalso:
    """Devuelve 1.0 por fila, o el valor fijado en 'respuestas' si se provee."""

    def __init__(self, respuestas: list[float] | None = None) -> None:
        self.respuestas = respuestas
        self.filas_recibidas: list[FeaturesCelda] = []

    def predecir_lote(self, filas):
        self.filas_recibidas.extend(filas)
        if self.respuestas is not None:
            return self.respuestas
        return [1.0] * len(filas)


class DistritoRepositorioFalso:
    def __init__(self, distritos: list[Distrito]) -> None:
        self._por_codigo = {d.codigo: d for d in distritos}

    def listar(self):
        return list(self._por_codigo.values())

    def obtener(self, codigo):
        return self._por_codigo.get(codigo)

    def existe(self, codigo):
        return codigo in self._por_codigo


class ClimaFalso:
    def normal_climatologica(self, distrito_codigo, mes):
        return 42.0


class CalendarioFalso:
    def es_feriado(self, fecha, distrito_codigo):
        return fecha.month == 8 and fecha.day == 6


class ExposicionFalso:
    def log_exposicion_v2(self, distrito_codigo, fecha):
        return 10.0


class TercilFalso:
    def tercil_y_confiabilidad(self, distrito_codigo):
        return Tercil.ALTO, 0.9775


_SAN_SALVADOR = Distrito(
    codigo="san salvador",
    nombre="San Salvador",
    departamento="San Salvador",
    pct_urbano=0.9965,
    poblacion=330543,
    poblacion_baja=False,
)
_ACAJUTLA = Distrito(
    codigo="acajutla",
    nombre="Acajutla",
    departamento="Sonsonate",
    pct_urbano=0.6622,
    poblacion=55307,
    poblacion_baja=False,
)


class TestPredecirCelda:
    def test_usa_prcp_enviado_por_el_cliente(self):
        predictor = PredictorFalso()
        caso_uso = PredecirCelda(
            predictor, DistritoRepositorioFalso([_SAN_SALVADOR]), ClimaFalso(), CalendarioFalso(), ExposicionFalso()
        )
        peticion = PeticionPrediccionCelda(
            distrito_codigo="san salvador", fecha=date(2024, 6, 15), franja=Franja.TARDE, prcp_mensual=99.0
        )
        resultado = caso_uso.ejecutar(peticion)

        assert resultado.fuente_prcp == FuentePrecipitacion.OBSERVADA
        assert predictor.filas_recibidas[0].prcp_mensual == 99.0

    def test_usa_normal_climatologica_si_no_se_envia_prcp(self):
        predictor = PredictorFalso()
        caso_uso = PredecirCelda(
            predictor, DistritoRepositorioFalso([_SAN_SALVADOR]), ClimaFalso(), CalendarioFalso(), ExposicionFalso()
        )
        peticion = PeticionPrediccionCelda(
            distrito_codigo="san salvador", fecha=date(2024, 6, 15), franja=Franja.TARDE
        )
        resultado = caso_uso.ejecutar(peticion)

        assert resultado.fuente_prcp == FuentePrecipitacion.NORMAL_CLIMATOLOGICA
        assert predictor.filas_recibidas[0].prcp_mensual == 42.0

    def test_periodo_agostino_y_feriado_el_6_de_agosto(self):
        predictor = PredictorFalso()
        caso_uso = PredecirCelda(
            predictor, DistritoRepositorioFalso([_SAN_SALVADOR]), ClimaFalso(), CalendarioFalso(), ExposicionFalso()
        )
        peticion = PeticionPrediccionCelda(
            distrito_codigo="san salvador", fecha=date(2024, 8, 6), franja=Franja.TARDE
        )
        caso_uso.ejecutar(peticion)

        fila = predictor.filas_recibidas[0]
        assert fila.periodo_agostino == 1
        assert fila.es_feriado == 1

    def test_distrito_desconocido_lanza_error_de_dominio(self):
        caso_uso = PredecirCelda(
            PredictorFalso(), DistritoRepositorioFalso([]), ClimaFalso(), CalendarioFalso(), ExposicionFalso()
        )
        peticion = PeticionPrediccionCelda(
            distrito_codigo="no_existe", fecha=date(2024, 1, 1), franja=Franja.TARDE
        )
        with pytest.raises(DistritoNoEncontradoError):
            caso_uso.ejecutar(peticion)


class TestRankingDistritos:
    def test_ordena_descendente_y_asigna_tercil(self):
        predictor = PredictorFalso(respuestas=None)
        caso_uso = RankingDistritos(
            predictor,
            DistritoRepositorioFalso([_SAN_SALVADOR, _ACAJUTLA]),
            ClimaFalso(),
            CalendarioFalso(),
            ExposicionFalso(),
            TercilFalso(),
        )
        peticion = PeticionRanking(fecha_inicio=date(2024, 1, 1), fecha_fin=date(2024, 1, 1))
        resultado = caso_uso.ejecutar(peticion)

        assert len(resultado) == 2
        assert resultado[0].tercil == Tercil.ALTO
        # 1 dia x 4 franjas x 1.0 por prediccion = 4.0 para cada distrito
        assert resultado[0].siniestros_esperados == pytest.approx(4.0)

    def test_una_fila_por_franja_por_dia(self):
        predictor = PredictorFalso()
        caso_uso = RankingDistritos(
            predictor,
            DistritoRepositorioFalso([_SAN_SALVADOR]),
            ClimaFalso(),
            CalendarioFalso(),
            ExposicionFalso(),
            TercilFalso(),
        )
        peticion = PeticionRanking(fecha_inicio=date(2024, 1, 1), fecha_fin=date(2024, 1, 2))
        caso_uso.ejecutar(peticion)

        # 2 dias x 4 franjas = 8 filas enviadas al predictor
        assert len(predictor.filas_recibidas) == 8
