"""Informe de siniestros esperados por periodo: caso de uso (con dobles en memoria) y
endpoints JSON/PDF/XLSX (TestClient contra la app real)."""

from __future__ import annotations

import csv
import io
import re
import math
import time
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from openpyxl import load_workbook

from app.application.calcular_equidad import ResultadoEquidad
from app.application.generar_informe_periodo import (
    GenerarInformePeriodo,
    PeticionInformeInvalidaError,
    PeticionInformePeriodo,
)
from app.application.predecir_celda import DistritoNoEncontradoError, PredecirCelda
from app.config import settings
from app.domain import poisson
from app.domain.informe import ComposicionVehiculos, PorcentajeVehiculo
from app.domain.models import ComposicionCausas, Franja, GrupoEquidad, ProporcionCausa
from app.application.generar_informe_periodo import repartir_terciles
from app.application.redactar_resumen import conteo
from app.domain.colores_causas import nombre_causa
from app.infrastructure.informes import formato
from tests.test_casos_uso import (
    _SAN_SALVADOR,
    CalendarioFalso,
    ClimaFalso,
    DistritoRepositorioFalso,
    ExposicionFalso,
    ParqueFalso,
)

_RELOJ = datetime(2026, 11, 2, 11, 45, tzinfo=ZoneInfo("America/El_Salvador"))
_NOMBRES_CAUSAS = [f"Causa {c}" for c in "ABCDEFGHIJ"]  # 10 causas -> 8 principales + «Otras causas»
_NOMBRES_PRESENTADOS = [nombre_causa(c) for c in _NOMBRES_CAUSAS]  # «Causa a»…


class PredictorDeterminista:
    """Esperado = (franja_cod + 1) × pct_urbano × (1 + 0.1 × dia_semana): distinto por distrito,
    franja y día de la semana."""

    def predecir_lote(self, filas):
        return [(f.franja_cod + 1) * f.pct_urbano * (1 + 0.1 * f.dia_semana) for f in filas]


class EquidadFalsa:
    def __init__(self, calibracion_rural: float) -> None:
        self._rural = calibracion_rural

    def ejecutar(self) -> ResultadoEquidad:
        return ResultadoEquidad(
            urbano_rural=[GrupoEquidad("Urbano", 0.97, 0.9, 50), GrupoEquidad("Rural", self._rural, 0.9, 50)],
            densidad_poblacional=[],
        )


def _distritos():
    return [
        replace(_SAN_SALVADOR, codigo=f"d{k}", nombre=f"Distrito {k}", pct_urbano=0.1 * k,
                departamento="Norte" if k % 2 else "Sur", poblacion=10_000 * k)
        for k in range(1, 8)
    ]


def _composicion() -> ComposicionCausas:
    por_franja = {}
    for franja in Franja:
        pesos = [10 - i - franja.codigo * 0.1 * (i % 3) for i in range(10)]
        total = sum(pesos)
        por_franja[franja] = tuple(ProporcionCausa(c, p / total) for c, p in zip(_NOMBRES_CAUSAS, pesos))
    return ComposicionCausas(date(2022, 1, 1), date(2024, 12, 31), {f: 100 for f in Franja}, por_franja)


class PredictorPlano:
    """Esperado constante: todos los días de la semana tienen el mismo promedio por día."""

    def predecir_lote(self, filas):
        return [1.0 for _ in filas]


def _caso_uso(calibracion_rural: float = 1.203, predictor=None) -> GenerarInformePeriodo:
    distritos = DistritoRepositorioFalso(_distritos())
    composicion = _composicion()
    predecir = PredecirCelda(
        predictor or PredictorDeterminista(), distritos, ClimaFalso(), CalendarioFalso(), ExposicionFalso(), ParqueFalso(), composicion
    )
    vehiculos = ComposicionVehiculos(
        date(2022, 1, 1), date(2024, 12, 31),
        (PorcentajeVehiculo("Automóvil", 49.24), PorcentajeVehiculo("Motocicleta", 21.07), PorcentajeVehiculo("Bus", 9.7)),
    )
    return GenerarInformePeriodo(
        predecir, distritos, composicion, vehiculos, EquidadFalsa(calibracion_rural), lambda: _RELOJ
    )


def _peticion(**kw) -> PeticionInformePeriodo:
    return PeticionInformePeriodo(**{"fecha_inicio": date(2026, 11, 2), "fecha_fin": date(2026, 11, 15), **kw})


class TestCasoDeUso:
    def test_totales_coinciden_en_todas_las_vistas(self):
        i = _caso_uso().ejecutar(_peticion())
        total = i.total.esperado
        assert sum(d.esperado for d in i.por_distrito) == pytest.approx(total)
        assert sum(f.esperado for f in i.por_franja) == pytest.approx(total)
        assert sum(d.suma for d in i.por_dia_semana) == pytest.approx(total)
        assert sum(c.esperado for c in i.por_causa) == pytest.approx(total)
        assert sum(sum(f.franjas.values()) for f in i.distrito_x_franja) == pytest.approx(total)
        assert sum(c.porcentaje for c in i.por_causa) == pytest.approx(100.0)

    def test_valor_esperado_conocido_de_un_distrito_y_un_dia(self):
        # d5: pct_urbano 0.5; 2026-11-02 es lunes (dia_semana 0): esperado por franja = (f+1) × 0.5
        i = _caso_uso().ejecutar(_peticion(fecha_fin=date(2026, 11, 2), distritos=("d5",)))
        assert [f.esperado for f in i.por_franja] == pytest.approx([0.5, 1.0, 1.5, 2.0])
        assert i.total.esperado == pytest.approx(5.0)

    def test_intervalos_son_poisson_del_esperado(self):
        i = _caso_uso().ejecutar(_peticion())
        assert (i.total.ic90_inf, i.total.ic90_sup) == poisson.intervalo_90(i.total.esperado)
        for d in i.por_distrito:
            assert (d.ic90_inf, d.ic90_sup) == poisson.intervalo_90(d.esperado)
        for c in i.por_causa:
            assert (c.ic90_inf, c.ic90_sup) == poisson.intervalo_90(c.esperado)

    def test_por_causa_son_8_mas_otras_ordenadas(self):
        i = _caso_uso().ejecutar(_peticion())
        causas = [c.causa for c in i.por_causa]
        assert len(causas) == 9 and causas[-1] == "Otras causas"
        principales = i.por_causa[:-1]
        assert [c.esperado for c in principales] == sorted((c.esperado for c in principales), reverse=True)
        assert set(causas[:-1]) <= set(_NOMBRES_PRESENTADOS)
        # Otras causas = las dos causas que quedaron fuera
        fuera = set(_NOMBRES_PRESENTADOS) - set(causas)
        assert len(fuera) == 2

    def test_causas_por_franja_suman_1_y_usan_las_mismas_causas(self):
        i = _caso_uso().ejecutar(_peticion())
        causas = [c.causa for c in i.por_causa]
        assert [x.franja for x in i.causas_por_franja] == list(Franja)
        for x in i.causas_por_franja:
            assert [c.causa for c in x.causas] == causas
            assert sum(c.proporcion for c in x.causas) == pytest.approx(1.0)

    def test_causas_por_franja_solo_incluye_las_franjas_filtradas(self):
        i = _caso_uso().ejecutar(_peticion(franjas=(Franja.NOCHE, Franja.MADRUGADA)))
        assert [x.franja for x in i.causas_por_franja] == [Franja.MADRUGADA, Franja.NOCHE]
        assert [f.esperado > 0 for f in i.por_franja] == [True, False, False, True]

    def test_el_informe_ya_no_tiene_causas_por_franja_distritos(self):
        i = _caso_uso().ejecutar(_peticion())
        assert not hasattr(i, "causas_por_franja_distritos")

    def test_distritos_del_filtro_ordenados_por_esperado(self):
        i = _caso_uso().ejecutar(_peticion(distritos=("d2", "d7", "d1")))
        assert [d.codigo for d in i.por_distrito] == ["d7", "d2", "d1"]  # por esperado desc

    def test_prioridades_son_las_10_con_mayor_promedio_por_dia(self):
        i = _caso_uso().ejecutar(_peticion(fecha_fin=date(2026, 11, 16)))  # 15 días: el lunes cae 3 veces
        assert len(i.prioridades) == 10
        esperados = [p.esperado for p in i.prioridades]
        assert esperados == sorted(esperados, reverse=True)
        # La mejor: d7 (pct_urbano 0.7), noche (×4), domingo (factor 1.6): promedio 4 × 0.7 × 1.6
        # por día; el domingo aparece 2 veces (8 y 15 de nov).
        mejor = i.prioridades[0]
        assert (mejor.distrito, mejor.franja, mejor.dia_semana, mejor.veces) == ("Distrito 7", Franja.NOCHE, "Domingo", 2)
        assert mejor.esperado == pytest.approx(4 * 0.7 * 1.6)
        # Cada prioridad es el promedio por día de su combinación y las 10 son las mayores de todas
        veces = {d.dia: d.veces for d in i.por_dia_semana}
        dias = list(veces)
        todas = sorted(
            ((franja.codigo + 1) * 0.1 * k * (1 + 0.1 * dow), f"Distrito {k}", franja, dias[dow])
            for k in range(1, 8) for franja in Franja for dow in range(7)
        )[::-1][:10]
        assert [round(e, 9) for e in esperados] == pytest.approx([round(t[0], 9) for t in todas])
        for pr in i.prioridades:
            assert pr.veces == veces[pr.dia_semana]

    def test_prioridades_se_ordenan_por_promedio_no_por_suma(self):
        # 15 días desde el lunes: el lunes aparece 3 veces y el resto 2. Con el predictor plano
        # todos los promedios empatan en 1.0 por celda: el orden NO debe favorecer al lunes
        # (más apariciones = más suma) sino resolverse por la regla de desempate estable.
        i = _caso_uso(predictor=PredictorPlano()).ejecutar(_peticion(fecha_fin=date(2026, 11, 16)))
        assert all(p.esperado == pytest.approx(1.0) for p in i.prioridades)
        # desempate: mayor suma (el lunes, 3 veces) y luego nombre; es determinista
        assert i.prioridades == _caso_uso(predictor=PredictorPlano()).ejecutar(_peticion(fecha_fin=date(2026, 11, 16))).prioridades
        assert i.prioridades[0].dia_semana == "Lunes"

    def test_dia_de_la_semana_es_promedio_por_dia_con_veces_y_suma(self):
        i = _caso_uso().ejecutar(_peticion(fecha_inicio=date(2026, 11, 2), fecha_fin=date(2026, 12, 2)))
        assert [d.veces for d in i.por_dia_semana] == [5, 5, 5, 4, 4, 4, 4]
        for d in i.por_dia_semana:
            assert d.esperado * d.veces == pytest.approx(d.suma)
        assert sum(d.suma for d in i.por_dia_semana) == pytest.approx(i.total.esperado)

    def test_dia_filtrado_tiene_veces_0_y_promedio_0(self):
        i = _caso_uso().ejecutar(_peticion(dias_semana=(2,)))
        miercoles = i.por_dia_semana[2]
        assert (miercoles.veces, miercoles.esperado > 0) == (2, True)
        otros = [d for k, d in enumerate(i.por_dia_semana) if k != 2]
        assert all((d.veces, d.esperado, d.suma) == (0, 0.0, 0.0) for d in otros)

    def test_dia_de_mayor_riesgo_no_depende_de_las_apariciones(self):
        # A.5. Mismo promedio por día en los 7 días, distinta cantidad de apariciones: ningún día
        # gana por aparecer 5 veces; el desempate es determinista (orden de la semana).
        inicio_lunes = _caso_uso(predictor=PredictorPlano()).ejecutar(
            _peticion(fecha_inicio=date(2026, 11, 2), fecha_fin=date(2026, 12, 2)))
        assert [d.veces for d in inicio_lunes.por_dia_semana] == [5, 5, 5, 4, 4, 4, 4]
        assert len({round(d.esperado, 9) for d in inicio_lunes.por_dia_semana}) == 1
        assert [d.suma for d in inicio_lunes.por_dia_semana][0] > [d.suma for d in inicio_lunes.por_dia_semana][3]
        # Empezando en jueves son jueves, viernes y sábado los que aparecen 5 veces: si ganara
        # «el de más apariciones» saldría uno de esos tres; sale el primero de la semana.
        inicio_jueves = _caso_uso(predictor=PredictorPlano()).ejecutar(
            _peticion(fecha_inicio=date(2026, 11, 5), fecha_fin=date(2026, 12, 5)))
        assert [d.veces for d in inicio_jueves.por_dia_semana] == [4, 4, 4, 5, 5, 5, 4]
        assert inicio_jueves.resumen.filas[2].valor.startswith("Lunes, ")
        assert inicio_jueves.resumen == _caso_uso(predictor=PredictorPlano()).ejecutar(
            _peticion(fecha_inicio=date(2026, 11, 5), fecha_fin=date(2026, 12, 5))).resumen
        # Con datos reales de promedio distinto gana el de mayor promedio aunque aparezca 4 veces
        # (jueves: factor 1.3, 4 veces; lunes: factor 1.0, 5 veces)
        i = _caso_uso().ejecutar(_peticion(fecha_inicio=date(2026, 11, 2), fecha_fin=date(2026, 12, 2),
                                           dias_semana=(0, 3)))
        assert [d.veces for d in i.por_dia_semana if d.veces] == [5, 4]
        assert i.resumen.filas[2].valor.startswith("Jueves, ")
        assert "los jueves son, en promedio, los días de mayor riesgo" in i.resumen.texto

    def test_terciles_se_recalculan_con_el_propio_informe(self):
        i = _caso_uso().ejecutar(_peticion())  # 7 distritos -> 3, 2, 2
        assert [d.tercil.value for d in i.por_distrito] == ["alto"] * 3 + ["medio"] * 2 + ["bajo"] * 2
        un = _caso_uso().ejecutar(_peticion(distritos=("d3",)))
        assert [d.tercil.value for d in un.por_distrito] == ["alto"]
        dos = _caso_uso().ejecutar(_peticion(distritos=("d3", "d4")))
        assert [d.tercil.value for d in dos.por_distrito] == ["alto", "medio"]

    @pytest.mark.parametrize(
        ("n", "esperado"),
        [(0, (0, 0, 0)), (1, (1, 0, 0)), (2, (1, 1, 0)), (3, (1, 1, 1)), (4, (2, 1, 1)), (5, (2, 2, 1)),
         (103, (35, 34, 34)), (104, (35, 35, 34))],
    )
    def test_reparto_de_terciles(self, n, esperado):
        terciles = [t.value for t in repartir_terciles(n)]
        assert (terciles.count("alto"), terciles.count("medio"), terciles.count("bajo")) == esperado
        assert terciles == sorted(terciles, key=["alto", "medio", "bajo"].index)  # alto primero

    def test_dias_de_la_semana_siempre_7_filas_de_lunes_a_domingo(self):
        i = _caso_uso().ejecutar(_peticion(dias_semana=(0, 6)))
        assert [d.dia for d in i.por_dia_semana] == [
            "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
        ]
        assert [d.esperado > 0 for d in i.por_dia_semana] == [True, False, False, False, False, False, True]
        assert [d.veces for d in i.por_dia_semana] == [2, 0, 0, 0, 0, 0, 2]
        assert i.supuestos.dias_con_lluvia_normal == 4  # 2 lunes + 2 domingos entre el 2 y el 15 de nov

    def test_supuestos_cuentan_los_dias_con_ultimo_parque(self):
        # ParqueFalso devuelve ULTIMO_DISPONIBLE a partir de 2027
        i = _caso_uso().ejecutar(_peticion(fecha_inicio=date(2026, 12, 30), fecha_fin=date(2027, 1, 2)))
        assert i.supuestos.dias_con_lluvia_normal == 4
        assert i.supuestos.dias_con_ultimo_parque == 2
        assert [v.tipo for v in i.supuestos.composicion_vehiculos] == ["Automóvil", "Motocicleta", "Bus"]

    def test_tasa_y_departamento(self):
        i = _caso_uso().ejecutar(_peticion(departamento="norte"))
        assert {d.departamento for d in i.por_distrito} == {"Norte"}
        d = i.por_distrito[0]
        assert d.tasa_10mil_hab == pytest.approx(d.esperado / int(d.codigo[1:]) / 10_000 * 10_000)

    def test_generado_usa_el_reloj_inyectado(self):
        assert _caso_uso().ejecutar(_peticion()).generado == _RELOJ

    def test_seis_limitaciones_con_cifras_calculadas(self):
        lim = _caso_uso().ejecutar(_peticion()).limitaciones
        assert len(lim) == 6
        assert lim[0].startswith("Datos sintéticos, calibrados sobre agregados de ONASEVI/FONAT")
        assert "(Automóvil 49.2%, Motocicleta 21.1%, Bus 9.7%)" in lim[3]
        assert lim[4] == (
            "En el periodo de evaluación el modelo sobreestimó un 20% los siniestros en los distritos "
            "rurales (menos de 30% de población urbana)."
        )
        assert lim[5].startswith("El rango del 90% refleja solo la variación aleatoria (Poisson)")
        assert "un 35%" in _caso_uso(1.35).ejecutar(_peticion()).limitaciones[4]
        assert "subestimó un 10%" in _caso_uso(0.9).ejecutar(_peticion()).limitaciones[4]

    def test_colores_incluyen_las_causas_y_otras_causas(self):
        i = _caso_uso().ejecutar(_peticion())
        assert {*_NOMBRES_PRESENTADOS, "Otras causas"} <= set(i.colores_causas)
        assert all(c.startswith("#") and len(c) == 7 for c in i.colores_causas.values())

    def test_nombres_de_causa_en_sentence_case_desde_una_funcion(self):
        assert nombre_causa("Distracción del Conductor") == "Distracción del conductor"
        assert nombre_causa("No Respetar Señales de Tránsito") == "No respetar señales de tránsito"
        assert nombre_causa("ESTADO DE EBRIEDAD O DROGA") == "Estado de ebriedad o droga"
        i = _caso_uso().ejecutar(_peticion())
        assert all(c.causa[0].isupper() and c.causa[1:] == c.causa[1:].lower() for c in i.por_causa)
        assert all(c.causa == nombre_causa(c.causa) for x in i.causas_por_franja for c in x.causas)

    @pytest.mark.parametrize(
        ("kw", "mensaje"),
        [
            ({"fecha_fin": date(2026, 11, 1)}, "anterior"),
            ({"fecha_fin": date(2027, 2, 2)}, "93 días"),
            ({"dias_semana": (7,)}, r"0 \(lunes\) a 6"),
            ({"departamento": "Atlantida"}, "Departamento desconocido"),
            ({"departamento": "Sur", "distritos": ("d1",)}, "no coinciden"),
            ({"fecha_fin": date(2026, 11, 2), "dias_semana": (3,)}, "Ningún día"),
        ],
    )
    def test_peticiones_invalidas(self, kw, mensaje):
        with pytest.raises(PeticionInformeInvalidaError, match=mensaje):
            _caso_uso().ejecutar(_peticion(**kw))

    def test_92_dias_es_el_maximo_admitido(self):
        _caso_uso().ejecutar(_peticion(fecha_fin=date(2027, 2, 1)))  # 92 días inclusive

    def test_distrito_inexistente(self):
        with pytest.raises(DistritoNoEncontradoError):
            _caso_uso().ejecutar(_peticion(distritos=("no-existe",)))

    def test_repetidos_en_los_filtros_se_descartan(self):
        i = _caso_uso().ejecutar(_peticion(distritos=("d1", "d1"), franjas=(Franja.TARDE, Franja.TARDE), dias_semana=(1, 1)))
        assert i.parametros.filtros.distritos == ("d1",)
        assert i.parametros.filtros.franjas == (Franja.TARDE,)
        assert i.parametros.filtros.dias_semana == (1,)


# ----------------------------------------------------------------------------- endpoints

_URL = "/api/v1/informes/periodo"
_PERIODO = {"fecha_inicio": "2026-11-02", "fecha_fin": "2026-12-02"}


def _fecha_mas(dias: int) -> str:
    return date.fromordinal(date(2026, 11, 2).toordinal() + dias).isoformat()


@pytest.fixture(scope="module")
def informe(client):
    r = client.post(_URL, json=_PERIODO)
    assert r.status_code == 200, r.text
    return r.json()


class TestEndpointJson:
    def test_claves_en_el_orden_del_contrato(self, informe):
        assert list(informe) == [
            "parametros", "generado", "resumen", "total", "por_distrito", "por_franja", "por_dia_semana",
            "distrito_x_franja", "por_causa", "causas_por_franja",
            "prioridades", "colores_causas", "supuestos", "limitaciones",
        ]
        assert list(informe["parametros"]["filtros"]) == ["departamento", "distritos", "franjas", "dias_semana"]
        assert list(informe["por_distrito"][0]) == [
            "codigo", "distrito", "departamento", "esperado", "ic90_inf", "ic90_sup", "tercil", "tasa_10mil_hab"
        ]
        assert list(informe["por_causa"][0]) == ["causa", "esperado", "ic90_inf", "ic90_sup", "porcentaje"]
        assert list(informe["prioridades"][0]) == ["distrito", "franja", "dia_semana", "esperado", "veces"]
        assert list(informe["por_dia_semana"][0]) == ["dia", "esperado", "veces", "suma"]
        assert list(informe["resumen"]) == ["texto", "filas"]
        assert "causas_por_franja_distritos" not in informe
        assert list(informe["supuestos"]) == ["dias_con_lluvia_normal", "dias_con_ultimo_parque", "composicion_vehiculos"]
        assert list(informe["causas_por_franja"][0]) == ["franja", "causas"]

    def test_totales_coinciden_en_todas_las_vistas(self, informe):
        total = informe["total"]["esperado"]
        assert len(informe["por_distrito"]) == 103
        for suma in (
            sum(d["esperado"] for d in informe["por_distrito"]),
            sum(f["esperado"] for f in informe["por_franja"]),
            sum(d["suma"] for d in informe["por_dia_semana"]),
            sum(c["esperado"] for c in informe["por_causa"]),
            sum(sum(d["franjas"].values()) for d in informe["distrito_x_franja"]),
        ):
            assert suma == pytest.approx(total, abs=1e-3)
        assert sum(c["porcentaje"] for c in informe["por_causa"]) == pytest.approx(100.0, abs=1e-3)

    def test_formas_del_contrato(self, informe):
        assert informe["generado"].endswith("-06:00")
        datetime.fromisoformat(informe["generado"])
        assert [f["franja"] for f in informe["por_franja"]] == [f.value for f in Franja]
        assert [c["causa"] for c in informe["por_causa"]][-1] == "Otras causas"
        assert len(informe["por_causa"]) == 9
        assert len(informe["prioridades"]) == 10
        assert all(isinstance(informe["total"][k], int) for k in ("ic90_inf", "ic90_sup"))
        assert {d["tercil"] for d in informe["por_distrito"]} <= {"alto", "medio", "bajo"}
        esperados = [d["esperado"] for d in informe["por_distrito"]]
        assert esperados == sorted(esperados, reverse=True)
        assert len(informe["colores_causas"]) == 18
        assert all(k == k[0].upper() + k[1:].lower() for k in informe["colores_causas"])  # sentence case
        assert informe["supuestos"]["dias_con_lluvia_normal"] == 31
        assert len(informe["limitaciones"]) == 6

    def test_dia_de_mayor_riesgo_usa_el_promedio_y_no_las_apariciones(self, informe):
        # 2026-11-02 (lunes) -> 2026-12-02: lunes, martes y miércoles aparecen 5 veces
        dias = informe["por_dia_semana"]
        assert [d["veces"] for d in dias] == [5, 5, 5, 4, 4, 4, 4]
        for d in dias:
            assert d["esperado"] * d["veces"] == pytest.approx(d["suma"], abs=1e-4)
        mejor = max(dias, key=lambda d: d["esperado"])
        assert mejor["dia"] not in ("Lunes", "Martes", "Miércoles")
        assert informe["resumen"]["filas"][2]["valor"].startswith(f"{mejor['dia']}, ")
        # el día con más SUMA es otro tipo de respuesta: aquí no debe decidir
        assert mejor["dia"] == max(dias, key=lambda d: d["esperado"])["dia"]

    def test_prioridades_por_promedio_y_tercil_35_34_34(self, informe):
        dias = {d["dia"]: d for d in informe["por_dia_semana"]}
        esperados = [p["esperado"] for p in informe["prioridades"]]
        assert esperados == sorted(esperados, reverse=True)
        assert all(p["veces"] == dias[p["dia_semana"]]["veces"] for p in informe["prioridades"])
        terciles = [d["tercil"] for d in informe["por_distrito"]]
        assert (terciles.count("alto"), terciles.count("medio"), terciles.count("bajo")) == (35, 34, 34)
        assert terciles == ["alto"] * 35 + ["medio"] * 34 + ["bajo"] * 34  # coincide con el orden de la tabla

    def test_resumen_con_cifras_reales(self, informe):
        resumen = informe["resumen"]
        total = informe["total"]
        texto = resumen["texto"]
        assert texto.startswith(
            f"Entre el 2 de noviembre y el 2 de diciembre de 2026 se esperan unos {conteo(total['esperado'])} "
            f"siniestros en el país (entre {total['ic90_inf']:,d} y {total['ic90_sup']:,d} con 90% de probabilidad). "
        )
        franja = max(informe["por_franja"], key=lambda f: f["esperado"])
        pct_franja = round(franja["esperado"] / total["esperado"] * 100)
        dias = informe["por_dia_semana"]
        dia = max(dias, key=lambda d: d["esperado"])["dia"].lower()
        nombre_franja = formato.etiqueta_franja(franja["franja"]).split(" (")[0].lower()
        assert f"La {nombre_franja} concentra el {pct_franja}% y los {dia}s son, en promedio, los días de mayor riesgo. " in texto
        top4 = informe["por_distrito"][:4]
        pct4 = round(sum(d["esperado"] for d in top4) / total["esperado"] * 100)
        assert f"{top4[0]['distrito']}, {top4[1]['distrito']}, {top4[2]['distrito']} y {top4[3]['distrito']} reúnen el {pct4}% del total." in texto
        por_franja = {x["franja"]: next(c["proporcion"] for c in x["causas"] if c["causa"] == "Estado de ebriedad o droga")
                      for x in informe["causas_por_franja"]}
        noche = round((por_franja["Noche (18-23)"] + por_franja["Madrugada (00-05)"]) / 2 * 100)
        dia_p = round((por_franja["Manana (06-11)"] + por_franja["Tarde (12-17)"]) / 2 * 100)
        assert texto.endswith(
            f"De noche y de madrugada, la ebriedad explica cerca del {noche}% de los siniestros esperados, frente al {dia_p}% del día."
        )
        assert "\n" not in texto and "•" not in texto
        assert [f["etiqueta"] for f in resumen["filas"]] == [
            "Total esperado", "Franja de mayor riesgo", "Día de mayor riesgo (promedio)",
            "Distrito con más siniestros esperados",
        ]
        assert resumen["filas"][0]["valor"] == f"{conteo(total['esperado'])} (entre {total['ic90_inf']:,d} y {total['ic90_sup']:,d})"
        assert resumen["filas"][1]["valor"].endswith(f", {pct_franja}% del total")
        assert resumen["filas"][3]["valor"] == f"{top4[0]['distrito']}, {top4[0]['esperado']:,.1f}"

    def test_proporciones_por_franja_suman_1(self, informe):
        for x in informe["causas_por_franja"]:
            assert sum(c["proporcion"] for c in x["causas"]) == pytest.approx(1.0)

    def test_limitaciones_calculadas_de_los_datos_reales(self, client, informe):
        rural = next(g for g in client.get("/api/v1/equidad").json()["urbano_rural"] if g["grupo"] == "Rural")
        pct = round((rural["calibracion"] - 1) * 100)
        assert f"sobreestimó un {pct}% los siniestros" in informe["limitaciones"][4]
        # Los porcentajes de vehículo salen de siniestros.csv 2022-2024, no de un valor escrito a mano
        conteo: dict[str, int] = {}
        with settings.ruta_siniestros.open(encoding="utf-8-sig") as fh:
            for fila in csv.DictReader(fh, delimiter=";"):
                if "2022-01-01" <= fila["fecha"] <= "2024-12-31":
                    conteo[fila["tipo_vehiculo"]] = conteo.get(fila["tipo_vehiculo"], 0) + 1
        total = sum(conteo.values())
        reales = {k: n / total * 100 for k, n in conteo.items()}
        vehiculos = informe["supuestos"]["composicion_vehiculos"]
        assert sum(v["porcentaje"] for v in vehiculos) == pytest.approx(100.0, abs=1e-3)
        for v in vehiculos:
            clave = v["tipo"].replace("ó", "o")  # el CSV escribe «Automovil», «Camion»
            assert v["porcentaje"] == pytest.approx(reales[clave], abs=1e-3)
        assert [v["porcentaje"] for v in vehiculos] == sorted((v["porcentaje"] for v in vehiculos), reverse=True)
        todos = ", ".join(f"{v['tipo']} {v['porcentaje']:.1f}%" for v in vehiculos)
        assert f"({todos})" in informe["limitaciones"][3]

    def test_rango_mayor_a_92_dias_devuelve_422(self, client):
        r = client.post(_URL, json={"fecha_inicio": "2026-11-02", "fecha_fin": _fecha_mas(92)})  # 93 días
        assert r.status_code == 422
        assert "92" in r.json()["detail"]
        assert client.post(_URL, json={"fecha_inicio": "2026-11-02", "fecha_fin": _fecha_mas(91)}).status_code == 200

    def test_validaciones(self, client):
        base = {"fecha_inicio": "2026-11-02", "fecha_fin": "2026-11-05"}
        assert client.post(_URL, json={**base, "fecha_fin": "2026-11-01"}).status_code == 422
        assert client.post(_URL, json={**base, "distritos": ["no-existe"]}).status_code == 404
        assert client.post(_URL, json={**base, "franjas": ["Mediodia"]}).status_code == 422
        assert client.post(_URL, json={**base, "dias_semana": [7]}).status_code == 422
        assert client.post(_URL, json={**base, "departamento": "Atlantida"}).status_code == 422
        assert client.post(_URL, json={"fecha_inicio": "2026-11-02"}).status_code == 422

    def test_filtros_se_aplican_y_se_reflejan(self, client):
        r = client.post(_URL, json={**_PERIODO, "departamento": "San Salvador", "franjas": ["Noche (18-23)"], "dias_semana": [4, 5]})
        assert r.status_code == 200
        i = r.json()
        assert i["parametros"]["filtros"] == {
            "departamento": "San Salvador", "distritos": [], "franjas": ["Noche (18-23)"], "dias_semana": [4, 5]
        }
        assert {d["departamento"] for d in i["por_distrito"]} == {"San Salvador"}
        assert [f["esperado"] > 0 for f in i["por_franja"]] == [False, False, False, True]
        assert [x["franja"] for x in i["causas_por_franja"]] == ["Noche (18-23)"]
        assert "concentra" not in i["resumen"]["texto"]  # una sola franja: se omite esa frase
        assert [d["veces"] for d in i["por_dia_semana"]] == [0, 0, 0, 0, 4, 4, 0]
        assert i["supuestos"]["dias_con_lluvia_normal"] == 8  # viernes y sábados entre el 2/11 y el 2/12

    def test_un_dia_un_distrito_coincide_con_la_tarjeta_del_mapa_de_riesgo(self, client):
        for codigo, fecha in (("san salvador", "2026-11-03"), ("santa ana", "2025-05-12")):
            informe = client.post(_URL, json={"fecha_inicio": fecha, "fecha_fin": fecha, "distritos": [codigo]}).json()
            tarjeta = client.post(
                "/api/v1/predicciones", json={"distritos": [codigo], "fecha_inicio": fecha, "fecha_fin": fecha}
            ).json()["predicciones"]
            for fila in informe["por_franja"]:
                celda = next(p for p in tarjeta if p["franja"] == fila["franja"])
                assert fila["esperado"] == pytest.approx(celda["siniestros_esperados"], abs=1e-5)
            # por_causa: mismas sumas por causa que las de las tarjetas (8 principales + resto)
            por_causa: dict[str, float] = {}
            for p in tarjeta:
                for c in p["por_causa"]:
                    clave = nombre_causa(c["causa"])  # la tarjeta usa el nombre del CSV
                    por_causa[clave] = por_causa.get(clave, 0.0) + c["esperado"]
            ranking = sorted(por_causa.items(), key=lambda kv: (-kv[1], kv[0]))
            esperado = [*ranking[:8], ("Otras causas", sum(v for _, v in ranking[8:]))]
            assert [c["causa"] for c in informe["por_causa"]] == [causa for causa, _ in esperado]
            assert [c["esperado"] for c in informe["por_causa"]] == pytest.approx([v for _, v in esperado], abs=1e-5)
            assert informe["total"]["esperado"] == pytest.approx(sum(p["siniestros_esperados"] for p in tarjeta), abs=1e-5)

    def test_rendimiento_92_dias_todos_los_distritos(self, client):
        t0 = time.perf_counter()
        r = client.post(_URL, json={"fecha_inicio": "2026-11-02", "fecha_fin": _fecha_mas(91)})
        segundos = time.perf_counter() - t0
        assert r.status_code == 200
        assert len(r.json()["por_distrito"]) == 103
        assert segundos < 5.0, f"92 días × 103 distritos tardó {segundos:.2f} s"


@pytest.fixture(autouse=True)
def _pypdf_sin_limite_de_tokens():
    """La portada incluye el mapa de distritos como vectores (cientos de trazos): el tope de
    seguridad de pypdf por flujo de contenido (2 millones) es demasiado bajo para extraer su texto."""
    pypdf = pytest.importorskip("pypdf")
    with pypdf.apply_configuration(stream_parsing_work_maximum_count=50_000_000):
        yield


class TestEndpointsPdfYExcel:
    _CASOS = [
        pytest.param({"fecha_inicio": "2026-11-03", "fecha_fin": "2026-11-03"}, id="1-dia"),
        pytest.param({**_PERIODO, "departamento": "La Libertad", "franjas": ["Tarde (12-17)", "Noche (18-23)"]}, id="30-dias-filtros"),
        pytest.param({"fecha_inicio": "2026-11-02", "fecha_fin": _fecha_mas(91)}, id="92-dias"),
        pytest.param({**_PERIODO, "distritos": ["san salvador", "santa ana"], "dias_semana": [0, 4]}, id="distritos-y-dias"),
    ]

    @staticmethod
    def _query(c: dict) -> list[tuple[str, str]]:
        q = [(k, v) for k, v in c.items() if isinstance(v, str)]
        q += [(k, str(x)) for k, v in c.items() if isinstance(v, list) for x in v]
        return q

    @pytest.mark.parametrize("caso", _CASOS)
    def test_pdf_se_genera_y_sus_cifras_coinciden_con_el_json(self, client, caso):
        pypdf = pytest.importorskip("pypdf")
        informe = client.post(_URL, json=caso).json()
        r = client.get(_URL + ".pdf", params=self._query(caso))
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.headers["content-disposition"] == (
            f'attachment; filename="informe-siniestros-{caso["fecha_inicio"]}_{caso["fecha_fin"]}.pdf"'
        )
        assert r.content.startswith(b"%PDF")

        lector = pypdf.PdfReader(io.BytesIO(r.content))
        assert 5 <= len(lector.pages) <= 25
        texto = "\n".join(p.extract_text() for p in lector.pages)
        texto_plano = " ".join(texto.split())
        # pypdf separa un par con kerning (p. ej. «7» + «.» de Public Sans: «67 .6»); el PDF se ve bien
        texto_plano = re.sub(r"(\d) ([.,]\d)", r"\1\2", texto_plano)
        # tamaño carta
        assert (float(lector.pages[0].mediabox.width), float(lector.pages[0].mediabox.height)) == (612.0, 792.0)
        total = informe["total"]
        resumen = informe["resumen"]
        assert " ".join(resumen["texto"].split()) in texto_plano  # el párrafo, tal cual
        for fila in resumen["filas"]:
            assert fila["etiqueta"] in texto_plano and fila["valor"] in texto_plano
        assert "Tabla 1. Resumen" in texto_plano
        for d in informe["por_distrito"]:  # Tabla 3: todos los distritos, un decimal
            assert d["distrito"] in texto_plano
            assert formato.numero(d["esperado"]) in texto_plano
        for p in informe["prioridades"]:  # Tabla 2: promedio por día
            assert formato.numero(p["esperado"]) in texto_plano
        for c in informe["por_causa"]:
            assert formato.numero(c["esperado"], 0) in texto_plano  # el rótulo del gráfico de causas
        for d in informe["por_dia_semana"]:
            assert f"{d['dia'].lower()} {d['veces']}" in texto_plano  # «Veces que aparece cada día…»
        for titulo in (
            "Figura 1. Siniestros esperados por causa en el periodo",
            "Figura 2. Composición de causas por franja horaria",
            "Figura 3. Siniestros esperados por día de la semana (promedio por día)",
            "Figura 4. Siniestros esperados por distrito y franja horaria",
            "Tabla 2.", "Tabla 3. Siniestros esperados por distrito",
        ):
            assert titulo in texto_plano
        assert texto_plano.count("Fuente: Radar Vial SV, modelo de frecuencia (NB12), con datos sintéticos calibrados a ONASEVI/FONAT.") == 7  # 4 figuras + 3 tablas
        # Supuestos y limitaciones: cada ítem completo y con su marcador (extracción plana de pypdf)
        seccion = texto[texto.index("Supuestos y limitaciones\n"):]
        seccion = " ".join(seccion.split())
        for n, limitacion in enumerate(informe["limitaciones"], 1):
            assert f"{n}. {' '.join(limitacion.split())}" in seccion
        assert "• La lluvia corresponde a la normal climatológica de cada mes." in seccion
        assert "• Los valores son promedios (esperados), no conteos observados." in seccion
        assert "Precipitación:" not in texto_plano
        # el aviso de datos sintéticos solo en el pie (una vez por página, no en la portada)
        aviso = "Datos sintéticos, calibrados sobre agregados de ONASEVI/FONAT. No usar para decisiones operativas"
        for pagina in lector.pages:
            assert pagina.extract_text().count(aviso) == 1
        assert "Página 2 de" in texto_plano  # la portada no lleva numeración (solo el aviso en el pie)
        assert "Página 1 de" not in texto_plano
        # lo eliminado
        assert "Son valores promedio: indican cómo suelen repartirse" not in texto_plano
        assert "Causa principal" not in texto_plano
        assert "Causas esperadas por franja en los distritos principales" not in texto_plano

    def test_pdf_incrusta_public_sans(self, client):
        pypdf = pytest.importorskip("pypdf")
        r = client.get(_URL + ".pdf", params={"fecha_inicio": "2026-11-03", "fecha_fin": "2026-11-03"})
        lector = pypdf.PdfReader(io.BytesIO(r.content))
        fuentes: dict[str, bool] = {}
        for pagina in lector.pages:
            for fuente in (pagina["/Resources"].get("/Font") or {}).values():
                fuente = fuente.get_object()
                descriptor = fuente["/DescendantFonts"][0].get_object()["/FontDescriptor"].get_object()
                fuentes[str(fuente["/BaseFont"])] = "/FontFile2" in descriptor
        assert fuentes, "el PDF no tiene fuentes"
        assert all(incrustada for incrustada in fuentes.values())
        assert all("Public-Sans" in nombre or "PublicSans" in nombre for nombre in fuentes), fuentes  # ninguna fuente de respaldo (Arial, etc.)

    @pytest.mark.parametrize("caso", _CASOS)
    def test_excel_se_genera_y_sus_cifras_coinciden_con_el_json(self, client, caso):
        informe = client.post(_URL, json=caso).json()
        r = client.get(_URL + ".xlsx", params=self._query(caso))
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert r.headers["content-disposition"] == (
            f'attachment; filename="informe-siniestros-{caso["fecha_inicio"]}_{caso["fecha_fin"]}.xlsx"'
        )
        libro = load_workbook(io.BytesIO(r.content))
        assert libro.sheetnames == [
            "Resumen", "Total", "Por distrito", "Por franja", "Por día de la semana", "Distrito x franja",
            "Por causa", "Causas por franja", "Prioridades", "Supuestos",
        ]
        total = libro["Total"]
        assert total["B2"].value == pytest.approx(informe["total"]["esperado"], abs=1e-5)
        assert (total["B3"].value, total["B4"].value) == (informe["total"]["ic90_inf"], informe["total"]["ic90_sup"])
        assert total["B2"].number_format == "#,##0.00"

        filas = list(libro["Por distrito"].iter_rows(min_row=2, values_only=True))
        assert len(filas) == len(informe["por_distrito"])
        for fila, d in zip(filas, informe["por_distrito"], strict=True):
            assert fila[:3] == (d["codigo"], d["distrito"], d["departamento"])
            assert fila[3] == pytest.approx(d["esperado"], abs=1e-5)
            assert fila[4:7] == (d["ic90_inf"], d["ic90_sup"], d["tercil"])
        causas = list(libro["Por causa"].iter_rows(min_row=2, values_only=True))
        assert [(f[0], f[2], f[3]) for f in causas] == [(c["causa"], c["ic90_inf"], c["ic90_sup"]) for c in informe["por_causa"]]
        assert [f[1] for f in causas] == pytest.approx([c["esperado"] for c in informe["por_causa"]], abs=1e-5)
        dias = list(libro["Por día de la semana"].iter_rows(min_row=1, values_only=True))
        assert dias[0] == ("Día", "Esperado por día (promedio)", "Veces en el periodo", "Suma en el periodo")
        for fila, d in zip(dias[1:], informe["por_dia_semana"], strict=True):
            assert fila[0] == d["dia"] and fila[2] == d["veces"]
            assert fila[1] == pytest.approx(d["esperado"], abs=1e-5) and fila[3] == pytest.approx(d["suma"], abs=1e-5)
        prioridades = list(libro["Prioridades"].iter_rows(min_row=1, values_only=True))
        assert prioridades[0][4:] == ("Esperado por día (promedio)", "Veces en el periodo")
        assert [f[5] for f in prioridades[1:]] == [p["veces"] for p in informe["prioridades"]]
        assert len(prioridades) == 11
        assert "Causas distritos" not in libro.sheetnames
        resumen = list(libro["Resumen"].iter_rows(values_only=True))
        assert resumen[1][0] == informe["resumen"]["texto"]
        assert [(f[0], f[1]) for f in resumen[4:8]] == [(x["etiqueta"], x["valor"]) for x in informe["resumen"]["filas"]]
        causas_excel = {f[1] for f in libro["Causas por franja"].iter_rows(min_row=2, values_only=True)}
        assert causas_excel <= set(informe["colores_causas"]) and "Estado de Ebriedad o Droga" not in causas_excel
        # encabezados con el color institucional, en negrita, y filas congeladas
        for hoja in libro.worksheets:
            assert hoja.freeze_panes == "A2"
        encabezado = libro["Por distrito"]["A1"]
        assert encabezado.font.bold and encabezado.fill.start_color.rgb.endswith("12355B")
        textos = [str(c.value) for fila in libro["Supuestos"].iter_rows() for c in fila if c.value]
        for limitacion in informe["limitaciones"]:
            assert any(limitacion in t for t in textos)
        assert "• La lluvia corresponde a la normal climatológica de cada mes." in textos
        assert not any("Precipitación" in t for t in textos)
        assert "Parámetros" in textos and caso["fecha_inicio"] in textos and caso["fecha_fin"] in textos

    def test_pdf_y_excel_validan_igual_que_el_json(self, client):
        for ext in (".pdf", ".xlsx"):
            largo = client.get(_URL + ext, params={"fecha_inicio": "2026-11-02", "fecha_fin": _fecha_mas(92)})
            assert largo.status_code == 422 and "92" in largo.json()["detail"]
            assert client.get(_URL + ext, params={"fecha_inicio": "2026-11-02", "fecha_fin": "2026-11-01"}).status_code == 422
            assert client.get(_URL + ext, params={"fecha_inicio": "2026-11-02"}).status_code == 422
            assert client.get(
                _URL + ext, params={"fecha_inicio": "2026-11-02", "fecha_fin": "2026-11-03", "distritos": "no-existe"}
            ).status_code == 404
            assert client.get(
                _URL + ext, params={"fecha_inicio": "2026-11-02", "fecha_fin": "2026-11-03", "franjas": "Mediodia"}
            ).status_code == 422
            assert client.get(
                _URL + ext, params={"fecha_inicio": "2026-11-02", "fecha_fin": "2026-11-03", "dias_semana": "9"}
            ).status_code == 422


class TestHtmlIntermedio:
    def test_html_con_estructura_pdf_y_sin_inyeccion(self, client):
        from app.infrastructure.informes.renderer_pdf import RendererPdf

        caso_uso = client.app.state.generar_informe_periodo_uc
        informe = caso_uso.ejecutar(PeticionInformePeriodo(date(2026, 11, 3), date(2026, 11, 3), distritos=("san salvador",)))
        html = RendererPdf().renderizar_html(informe)
        assert html.count("<svg") == 6  # logotipo + mapa de la portada + 2 de causas + día de la semana + mapa de calor
        for rotulo in ("Figura 1.", "Figura 2.", "Figura 3.", "Figura 4.", "Tabla 1.", "Tabla 2.", "Tabla 3."):
            assert rotulo in html
        assert "text-transform" not in RendererPdf().renderizar_html(informe) and "Causa principal" not in html
        assert "Radar Vial SV" in html and "Informe de siniestros esperados" in html
        assert "Dónde y cuándo concentrar los controles" in html
        assert math.isclose(informe.total.esperado, sum(d.esperado for d in informe.por_distrito))
