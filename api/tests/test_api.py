"""Tests end-to-end de la API (httpx TestClient contra la app real, con los
artefactos versionados del repositorio)."""

from __future__ import annotations

import json
import time

import pytest

from app.config import settings

from app.infrastructure.geo.geojson_adapter import GeoJsonAdapter
from app.infrastructure.web.dependencias import get_geojson_adapter
from app.main import app


_NO_MONOTONO = pytest.mark.xfail(
    strict=True,
    reason="Hallazgo: el modelo XGBoost no es monótono en prcp_mensual (sin monotone_constraints). "
    "Con 2024-06-15/Tarde, +100% baja hasta -8.7% en 48 distritos y -100% sube hasta +19.9% "
    "(Mejicanos) en 8. Requiere decisión: restricción monótona al reentrenar o aceptar y explicar.",
)


class TestSalud:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["estado"] == "ok"
        assert body["modelo_cargado"] is True
        assert body["prediccion_canaria_ok"] is True
        assert body["log_tasa_base"] == -12.936335060885538
        assert body["version_modelo"] == "frecuencia_xgboost_offset_v3.pkl"


class TestModeloInfo:
    """Contrato de GET /api/v1/modelo/info; todo valor se compara contra los JSON de NB12."""

    @pytest.fixture(scope="module")
    def jsons(self):
        espec = json.loads(settings.ruta_especificacion_nb12.read_text(encoding="utf-8"))
        metricas = json.loads(settings.ruta_metricas_nb12.read_text(encoding="utf-8"))
        return espec, metricas

    def test_identidad_del_modelo(self, client, jsons):
        espec, _ = jsons
        body = client.get("/api/v1/modelo/info").json()
        assert body["version_modelo"] == espec["modelo"]
        assert body["fecha_entrenamiento"] == espec["generado"]
        assert body["variables"] == espec["columnas"]
        assert len(body["variables"]) == 12
        assert "log_tasa_base" in body["offset"]
        assert "NB07" not in body["nombre"]

    def test_metricas_final_y_referencia_nb07(self, client, jsons):
        _, m = jsons
        body = client.get("/api/v1/modelo/info").json()
        assert body["metricas"] == m["prueba"]["comparacion"]["ampliado (NB12)"]
        assert body["metricas_referencia_nb07"] == m["prueba"]["comparacion"]["NB07"]
        assert body["metricas"]["calibracion"] == pytest.approx(0.9765565538188802)

    def test_pruebas_de_significancia(self, client, jsons):
        _, m = jsons
        pruebas = client.get("/api/v1/modelo/info").json()["pruebas_significancia"]
        t = m["prueba"]["prueba_mensual"]
        assert pruebas["t_pareada_mensual"] == {
            "n_meses": t["n_meses"],
            "t": t["t"],
            "p": t["p_t"],
            "meses_que_mejora": t["meses_que_mejora"],
        }
        dm = m["prueba"]["diebold_mariano"]
        assert pruebas["diebold_mariano"] == {"DM": dm["DM"], "p": dm["p"], "dias": dm["dias"]}

    def test_importancia_shap_ordenada_descendente(self, client, jsons):
        _, m = jsons
        shap = client.get("/api/v1/modelo/info").json()["importancia_shap"]
        valores = [f["valor"] for f in shap]
        assert valores == sorted(valores, reverse=True)
        assert {f["variable"]: f["valor"] for f in shap} == m["importancia_shap"]
        assert shap[0]["variable"] == "franja_cod"

    def test_calibracion_por_franja(self, client, jsons):
        _, m = jsons
        franjas = client.get("/api/v1/modelo/info").json()["calibracion_por_franja"]
        esperado = m["prueba"]["calibracion_por_franja"]["ampliado"]
        assert {f["franja"]: f["calibracion"] for f in franjas} == esperado
        assert len(franjas) == 4

    def test_seis_limitaciones_exactas(self, client):
        limitaciones = client.get("/api/v1/modelo/info").json()["limitaciones"]
        assert len(limitaciones) == 6
        assert limitaciones[2] == (
            "En el periodo de evaluación (2025–2026) el modelo sobreestima un 20% los "
            "siniestros en los distritos rurales (menos de 30% de población urbana) y un "
            "40% en la madrugada (00:00 a 05:59)."
        )
        assert limitaciones[3] == (
            "El ordenamiento de los distritos por riesgo es algo menos preciso que el del "
            "modelo anterior (correlación de Spearman de 0.945 frente a 0.971)."
        )
        assert limitaciones[4] == (
            "Para fechas fuera del periodo de enero de 2022 a junio de 2026, el crecimiento "
            "del parque vehicular usa el último mes disponible."
        )
        assert limitaciones[5].startswith("El submodelo de severidad no se expone")

    def test_cifras_de_las_limitaciones_coinciden_con_los_json(self, jsons):
        _, m = jsons
        assert round((m["equidad"]["prueba 2025-2026 | rural (pct_urbano < 0.3)"]["ampliado"] - 1) * 100) == 20
        assert round((m["prueba"]["calibracion_por_franja"]["ampliado"]["Madrugada (00-05)"] - 1) * 100) == 40
        assert round(m["prueba"]["comparacion"]["ampliado (NB12)"]["spearman_global"], 3) == 0.945
        assert round(m["prueba"]["comparacion"]["NB07"]["spearman_global"], 3) == 0.971


class TestDistritos:
    def test_lista_103_distritos(self, client):
        r = client.get("/api/v1/distritos")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 103
        assert len(body["distritos"]) == 103

    def test_nombres_para_mostrar_con_tildes(self, client):
        por_codigo = {d["codigo"]: d for d in client.get("/api/v1/distritos").json()["distritos"]}
        assert por_codigo["antiguo cuscatlan"]["nombre"] == "Antiguo Cuscatlán"
        assert por_codigo["villa dolores"]["nombre"] == "Villa Dolores"

    def test_geojson_devuelve_200_con_las_geometrias_reales(self, client):
        # El repositorio ya tiene un GeoJSON real colocado en
        # data/geo/distritos.geojson (ver README §7) — este test cubre el
        # camino feliz contra ese archivo real, versionado.
        r = client.get("/api/v1/distritos/geojson")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) > 0

    def test_geojson_devuelve_503_si_el_archivo_no_existe(self, tmp_path):
        # No depende de que el archivo real del repo esté ausente (dejó de
        # estarlo — ver README §7): se aísla con un dependency override,
        # igual que tests/test_geojson_adapter.py hace a nivel de adaptador.
        from fastapi.testclient import TestClient

        app.dependency_overrides[get_geojson_adapter] = lambda: GeoJsonAdapter(
            tmp_path / "no_existe.geojson"
        )
        try:
            with TestClient(app) as client_aislado:
                r = client_aislado.get("/api/v1/distritos/geojson")
        finally:
            del app.dependency_overrides[get_geojson_adapter]

        assert r.status_code == 503
        assert "geojson" in r.json()["detail"].lower()


class TestPredicciones:
    def test_prediccion_valida(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-15",
                "franjas": ["Tarde (12-17)"],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "aviso" in body and "sintético" in body["aviso"]
        assert len(body["predicciones"]) == 1
        celda = body["predicciones"][0]
        assert celda["distrito_codigo"] == "san salvador"
        assert celda["fuente_prcp"] == "normal_climatologica"
        assert 0 < celda["siniestros_esperados"] < 20
        assert celda["fuente_parque"] == "observado"

    @pytest.mark.parametrize("distrito", ["san salvador", "soyapango", "santa tecla"])
    def test_300_mm_predice_mas_que_0_mm_en_el_area_metropolitana(self, client, distrito):
        def esperado(prcp):
            r = client.post(
                "/api/v1/predicciones",
                json={
                    "distritos": [distrito],
                    "fecha_inicio": "2024-06-15",
                    "fecha_fin": "2024-06-15",
                    "franjas": ["Tarde (12-17)"],
                    "prcp_mensual": prcp,
                },
            )
            assert r.status_code == 200
            return r.json()["predicciones"][0]["siniestros_esperados"]

        assert esperado(300) > esperado(0)

    @staticmethod
    def _por_distrito(client, **extra):
        codigos = [d["codigo"] for d in client.get("/api/v1/distritos").json()["distritos"]]
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": codigos,
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-15",
                "franjas": ["Tarde (12-17)"],
                **extra,
            },
        )
        assert r.status_code == 200
        return {c["distrito_codigo"]: c for c in r.json()["predicciones"]}

    @_NO_MONOTONO
    def test_escenario_mas_100_pct_nunca_reduce_ningun_distrito(self, client):
        base = self._por_distrito(client)
        doble = self._por_distrito(client, factor_prcp=1)
        assert len(base) == 103
        bajaron = [d for d in base if doble[d]["siniestros_esperados"] < base[d]["siniestros_esperados"] - 1e-9]
        assert bajaron == []
        assert {c["fuente_prcp"] for c in doble.values()} == {"escenario_relativo"}

    @_NO_MONOTONO
    def test_escenario_menos_100_pct_nunca_aumenta_ningun_distrito(self, client):
        base = self._por_distrito(client)
        seco = self._por_distrito(client, factor_prcp=-1)
        subieron = [d for d in base if seco[d]["siniestros_esperados"] > base[d]["siniestros_esperados"] + 1e-9]
        assert subieron == []
        assert all(c["prcp_mensual"] == 0 for c in seco.values())

    def test_factor_prcp_aplica_la_normal_de_cada_distrito_y_respeta_el_rango(self, client):
        base = self._por_distrito(client)
        doble = self._por_distrito(client, factor_prcp=1)
        for d, c in base.items():
            assert doble[d]["prcp_mensual"] == pytest.approx(min(812.1, c["prcp_mensual"] * 2))
        assert len({round(c["prcp_mensual"], 3) for c in doble.values()}) > 1

    def test_factor_prcp_y_prcp_mensual_son_excluyentes(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-15",
                "prcp_mensual": 100,
                "factor_prcp": 0.5,
            },
        )
        assert r.status_code == 422

    @pytest.mark.parametrize("factor", [-1.5, 1.5])
    def test_factor_prcp_fuera_de_rango_es_422(self, client, factor):
        r = client.post(
            "/api/v1/predicciones",
            json={"distritos": ["san salvador"], "fecha_inicio": "2024-06-15", "fecha_fin": "2024-06-15", "factor_prcp": factor},
        )
        assert r.status_code == 422

    def test_las_cuatro_franjas_dan_valores_distintos_y_madrugada_es_la_menor(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={"distritos": ["san salvador"], "fecha_inicio": "2024-06-15", "fecha_fin": "2024-06-15"},
        )
        valores = {c["franja"]: c["siniestros_esperados"] for c in r.json()["predicciones"]}
        assert len(valores) == 4
        assert len(set(valores.values())) == 4
        assert min(valores, key=valores.get) == "Madrugada (00-05)"

    def test_fecha_fuera_de_rango_del_parque_declara_ultimo_disponible(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2027-03-01",
                "fecha_fin": "2027-03-01",
                "franjas": ["Tarde (12-17)"],
            },
        )
        assert r.status_code == 200
        assert r.json()["predicciones"][0]["fuente_parque"] == "ultimo_disponible"

    def test_un_mes_completo_de_todos_los_distritos_tarda_menos_de_3_segundos(self, client):
        codigos = [d["codigo"] for d in client.get("/api/v1/distritos").json()["distritos"]]
        assert len(codigos) == 103
        inicio = time.perf_counter()
        r = client.post(
            "/api/v1/predicciones",
            json={"distritos": codigos, "fecha_inicio": "2025-06-01", "fecha_fin": "2025-06-30"},
        )
        duracion = time.perf_counter() - inicio
        assert r.status_code == 200
        assert len(r.json()["predicciones"]) == 103 * 30 * 4
        assert duracion < 3.0, f"un mes completo tardó {duracion:.2f} s"

    def test_prediccion_con_prcp_enviado(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-15",
                "prcp_mensual": 50.0,
            },
        )
        assert r.status_code == 200
        for celda in r.json()["predicciones"]:
            assert celda["fuente_prcp"] == "observada"

    def test_distrito_inexistente_devuelve_404(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["no_existe"],
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-15",
            },
        )
        assert r.status_code == 404

    def test_fecha_absurda_devuelve_422(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "1500-01-01",
                "fecha_fin": "1500-01-01",
            },
        )
        assert r.status_code == 422

    def test_fecha_fin_antes_de_inicio_devuelve_422(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-01",
            },
        )
        assert r.status_code == 422

    def test_franja_invalida_devuelve_422(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2024-06-15",
                "fecha_fin": "2024-06-15",
                "franjas": ["Franja inexistente"],
            },
        )
        assert r.status_code == 422

    def test_rango_demasiado_largo_devuelve_422(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={
                "distritos": ["san salvador"],
                "fecha_inicio": "2022-01-01",
                "fecha_fin": "2024-01-01",
            },
        )
        assert r.status_code == 422


class TestRanking:
    def test_ranking_devuelve_103_distritos_con_tercil(self, client):
        r = client.get(
            "/api/v1/ranking",
            params={"fecha_inicio": "2024-06-01", "fecha_fin": "2024-06-07"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["ranking"]) == 103
        for item in body["ranking"]:
            assert item["tercil"] in ("alto", "medio", "bajo")
            assert 0 <= item["confiabilidad_spearman"] <= 1

    def test_ranking_esta_ordenado_descendente(self, client):
        r = client.get(
            "/api/v1/ranking",
            params={"fecha_inicio": "2024-06-01", "fecha_fin": "2024-06-07"},
        )
        valores = [item["siniestros_esperados"] for item in r.json()["ranking"]]
        assert valores == sorted(valores, reverse=True)

    def test_ranking_fecha_invalida_devuelve_422(self, client):
        r = client.get(
            "/api/v1/ranking",
            params={"fecha_inicio": "2024-06-07", "fecha_fin": "2024-06-01"},
        )
        assert r.status_code == 422


class TestImpactoLey:
    def test_serie_anual_real_y_fecha_de_ley(self, client):
        r = client.get("/api/v1/impacto-ley-cero-tolerancia")
        assert r.status_code == 200
        body = r.json()
        assert body["fecha_aprobacion_ley"] == "2024-12-20"
        anio_2025 = next(f for f in body["serie_anual"] if f["anio"] == "2025")
        assert anio_2025["siniestros"] == 22265
        assert anio_2025["fallecidos"] == 1238
        assert len(body["limitaciones"]) > 0
        assert not any("no medido" in l for l in body["limitaciones"])

    def test_tasas_por_vehiculo_con_parque_real(self, client):
        tasas = {t["anio"]: t for t in client.get("/api/v1/impacto-ley-cero-tolerancia").json()["tasas_por_vehiculo"]}
        assert list(tasas) == ["2022", "2023", "2024", "2025", "2026"]
        assert tasas["2022"]["siniestros_por_1000_vehiculos"] == pytest.approx(10.9, abs=0.05)
        assert tasas["2025"]["fallecidos_por_10000_vehiculos"] == pytest.approx(6.13, abs=0.01)
        assert all(t["parque_es_dato_real"] is True for t in tasas.values())
        assert tasas["2026"]["parcial"] is True

    def test_resumen_expone_null_explicito_en_el_periodo_parcial(self, client):
        resumen = client.get("/api/v1/impacto-ley-cero-tolerancia").json()["resumen"]
        parcial = next(f for f in resumen if f["metrica"] == "siniestros" and f["periodo"] == "2026 ene-jun")
        assert "variacion_interanual_pct" in parcial
        assert parcial["variacion_interanual_pct"] is None

    def test_resumen_trae_variacion_interanual(self, client):
        r = client.get("/api/v1/impacto-ley-cero-tolerancia")
        resumen = r.json()["resumen"]
        fila = next(f for f in resumen if f["metrica"] == "siniestros" and f["periodo"] == "2025")
        assert fila["variacion_interanual_pct"] == pytest.approx(9.674400275848472, abs=1e-6)


class TestUsuariosVulnerables:
    def test_reparto_coincide_con_nb09(self, client):
        r = client.get("/api/v1/usuarios-vulnerables")
        assert r.status_code == 200
        body = r.json()
        peaton = next(t for t in body["reparto_por_tipo"] if t["tipo"] == "Peatón")
        assert peaton["pct_fatales"] == pytest.approx(42.47, abs=0.01)
        assert peaton["letalidad_por_siniestro"] == pytest.approx(0.2294, abs=0.0001)

    def test_letalidad_por_edad_creciente_en_los_extremos(self, client):
        r = client.get("/api/v1/usuarios-vulnerables")
        por_edad = {f["rango_etario"]: f["letalidad"] for f in r.json()["letalidad_por_edad"]}
        assert por_edad["71+"] == pytest.approx(0.2982, abs=0.0001)
        assert por_edad["0-15"] == pytest.approx(0.0900, abs=0.0001)


class TestComparacionHistorica:
    def test_devuelve_los_103_distritos_con_brecha(self, client):
        r = client.get("/api/v1/comparacion-historica")
        assert r.status_code == 200
        body = r.json()
        assert len(body["distritos"]) == 103
        for d in body["distritos"]:
            assert d["observado"] >= 0
            assert d["predicho"] >= 0

    def test_esta_ordenado_por_observado_descendente(self, client):
        r = client.get("/api/v1/comparacion-historica")
        valores = [d["observado"] for d in r.json()["distritos"]]
        assert valores == sorted(valores, reverse=True)


class TestEquidad:
    def test_devuelve_los_dos_cortes(self, client):
        r = client.get("/api/v1/equidad")
        assert r.status_code == 200
        body = r.json()
        grupos_urbano_rural = {g["grupo"] for g in body["urbano_rural"]}
        grupos_densidad = {g["grupo"] for g in body["densidad_poblacional"]}
        assert grupos_urbano_rural == {"Urbano", "Rural"}
        assert grupos_densidad == {"Alta densidad", "Baja densidad"}
        for g in body["urbano_rural"] + body["densidad_poblacional"]:
            assert g["n_distritos"] > 0

    def test_rural_es_predicho_sobre_observado_y_coincide_con_nb12(self, client):
        metricas = json.loads(settings.ruta_metricas_nb12.read_text(encoding="utf-8"))
        esperado = metricas["equidad"]["prueba 2025-2026 | rural (pct_urbano < 0.3)"]["ampliado"]
        rural = next(g for g in client.get("/api/v1/equidad").json()["urbano_rural"] if g["grupo"] == "Rural")
        assert rural["calibracion"] == pytest.approx(esperado, abs=0.005)
        assert rural["calibracion"] == pytest.approx(1.20, abs=0.01)


class TestCors:
    def test_origen_por_defecto_permitido_y_sin_credenciales(self, client):
        r = client.options(
            "/api/v1/distritos",
            headers={"Origin": "http://localhost:4200", "Access-Control-Request-Method": "GET"},
        )
        assert r.headers["access-control-allow-origin"] == "http://localhost:4200"
        assert "access-control-allow-credentials" not in r.headers

    def test_origen_desconocido_no_se_permite(self, client):
        r = client.options(
            "/api/v1/distritos",
            headers={"Origin": "http://malicioso.example", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in r.headers


class TestSiniestrosPuntos:
    def test_sin_filtro_devuelve_los_89946_puntos(self, client):
        # 89,946 < el tope de 90,000 por respuesta: sin filtro cabe entero.
        r = client.get("/api/v1/siniestros/puntos")
        assert r.status_code == 200
        assert r.json()["total"] == 89_946

    def test_filtra_por_tipo_siniestro(self, client):
        r = client.get("/api/v1/siniestros/puntos", params={"tipo_siniestro": "Volcamiento"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1656
        assert len(body["puntos"]) == 1656
        assert all(p["tipo_siniestro"] == "Volcamiento" for p in body["puntos"])
        assert "aviso" in body

    def test_filtra_por_rango_de_fechas(self, client):
        r = client.get(
            "/api/v1/siniestros/puntos",
            params={"fecha_inicio": "2022-01-01", "fecha_fin": "2022-01-01"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] > 0
        assert all(p["fecha"] == "2022-01-01" for p in body["puntos"])

    def test_fecha_fin_anterior_a_fecha_inicio_da_422(self, client):
        r = client.get(
            "/api/v1/siniestros/puntos",
            params={"fecha_inicio": "2022-01-05", "fecha_fin": "2022-01-01"},
        )
        assert r.status_code == 422

    def test_hora_con_formato_invalido_da_422(self, client):
        r = client.get("/api/v1/siniestros/puntos", params={"hora_inicio": "9am"})
        assert r.status_code == 422

    def test_puntos_incluyen_coordenadas_reales_de_el_salvador(self, client):
        r = client.get("/api/v1/siniestros/puntos", params={"departamento": "Cabañas"})
        assert r.status_code == 200
        for p in r.json()["puntos"][:20]:
            assert 12.9 <= p["latitud"] <= 14.5
            assert -90.2 <= p["longitud"] <= -87.6


class TestSiniestrosOpcionesFiltro:
    def test_devuelve_las_opciones_reales(self, client):
        r = client.get("/api/v1/siniestros/opciones-filtro")
        assert r.status_code == 200
        body = r.json()
        assert len(body["departamentos"]) == 14
        assert "San Salvador" in body["departamentos"]
        assert set(body["tipos_siniestro"]) == {
            "Colisión", "Atropello", "Características especiales",
            "Choque contra objeto fijo", "Volcamiento",
        }


class TestClima:
    def test_rango_entrenamiento_viene_de_la_matriz(self, client):
        r = client.get("/api/v1/clima/rango-entrenamiento")
        assert r.status_code == 200
        assert r.json() == {"minimo_mm": 0.0, "maximo_mm": 812.1, "anio_inicio": 2022, "anio_fin": 2024}


class TestAgregacionSiniestros:
    """Agregaciones de /api/v1/siniestros/puntos (hexágonos y distritos)."""

    TOTAL = 89_946

    @staticmethod
    def _agregar(client, agregacion, **filtros):
        r = client.get("/api/v1/siniestros/puntos", params={"agregacion": agregacion, **filtros})
        assert r.status_code == 200, r.text
        return r.json()

    def test_con_filtros_vacios_la_suma_de_hexagonos_es_el_total_de_siniestros(self, client):
        cuerpo = self._agregar(client, "hex")
        assert sum(h["conteo"] for h in cuerpo["hexagonos"]) == self.TOTAL
        assert cuerpo["total"] == self.TOTAL
        assert cuerpo["agregacion"] == "hex"
        assert cuerpo["lado_km"] == 2.0

    def test_con_filtros_vacios_la_suma_por_distrito_es_el_total_de_siniestros(self, client):
        cuerpo = self._agregar(client, "distrito")
        assert sum(d["siniestros"] for d in cuerpo["distritos"]) == self.TOTAL
        assert cuerpo["total"] == self.TOTAL
        assert len(cuerpo["distritos"]) == 103

    @pytest.mark.parametrize(
        "filtros",
        [
            {"fecha_inicio": "2025-01-01", "fecha_fin": "2025-12-31"},
            {"tipo_siniestro": "Colisión", "departamento": "San Salvador"},
            {"hora_inicio": "06:00", "hora_fin": "11:59", "tipo_vehiculo": "Motocicleta"},
        ],
    )
    def test_con_un_mismo_filtro_el_total_de_las_dos_agregaciones_coincide(self, client, filtros):
        hexagonos = self._agregar(client, "hex", **filtros)
        distritos = self._agregar(client, "distrito", **filtros)
        puntos = client.get("/api/v1/siniestros/puntos", params=filtros).json()
        assert hexagonos["total"] == distritos["total"] == puntos["total"] > 0

    def test_hexagonos_traen_conteo_distritos_y_seis_vertices(self, client):
        h = self._agregar(client, "hex", departamento="San Salvador")["hexagonos"][0]
        assert h["conteo"] > 0
        assert h["distritos"] and all(isinstance(d, str) for d in h["distritos"])
        assert len(h["vertices"]) == 6
        assert 13 < h["centro"][0] < 15 and -91 < h["centro"][1] < -87

    def test_distritos_traen_poblacion_y_tasa_por_10mil(self, client):
        d = next(x for x in self._agregar(client, "distrito")["distritos"] if x["distrito_codigo"] == "san salvador")
        assert d["departamento"] == "San Salvador"
        assert d["poblacion"] > 0
        assert d["tasa_por_10mil"] == pytest.approx(d["siniestros"] / d["poblacion"] * 10_000)

    def test_un_filtro_sin_resultados_da_ceros_y_mantiene_los_103_distritos(self, client):
        cuerpo = self._agregar(client, "distrito", fecha_inicio="2030-01-01")
        assert cuerpo["total"] == 0
        assert len(cuerpo["distritos"]) == 103
        assert self._agregar(client, "hex", fecha_inicio="2030-01-01")["hexagonos"] == []

    def test_las_agregaciones_no_tienen_el_tope_de_90000_puntos(self, client):
        # Sin filtros el endpoint de puntos devuelve el máximo; las agregaciones siempre responden.
        assert self._agregar(client, "hex")["total"] == self.TOTAL

    def test_agregacion_desconocida_es_422(self, client):
        r = client.get("/api/v1/siniestros/puntos", params={"agregacion": "circulo"})
        assert r.status_code == 422

    def test_sin_agregacion_sigue_devolviendo_puntos(self, client):
        cuerpo = client.get("/api/v1/siniestros/puntos", params={"departamento": "Cabañas", "fecha_inicio": "2025-01-01", "fecha_fin": "2025-01-31"}).json()
        assert "puntos" in cuerpo and "hexagonos" not in cuerpo


class TestComposicionCausas:
    FRANJAS = ["Madrugada (00-05)", "Manana (06-11)", "Tarde (12-17)", "Noche (18-23)"]

    def test_endpoint_devuelve_las_cuatro_franjas_con_proporciones_que_suman_1(self, client):
        cuerpo = client.get("/api/v1/composicion-causas").json()
        assert cuerpo["periodo_inicio"] == "2022-01-01" and cuerpo["periodo_fin"] == "2024-12-31"
        assert [f["franja"] for f in cuerpo["franjas"]] == self.FRANJAS
        for f in cuerpo["franjas"]:
            assert sum(c["proporcion"] for c in f["causas"]) == pytest.approx(1.0)
            proporciones = [c["proporcion"] for c in f["causas"]]
            assert proporciones == sorted(proporciones, reverse=True)
            assert f["n_siniestros"] > 0

    def test_los_nombres_de_causa_son_los_del_filtro_de_hotspots(self, client):
        opciones = set(client.get("/api/v1/siniestros/opciones-filtro").json()["factores_causantes"])
        for f in client.get("/api/v1/composicion-causas").json()["franjas"]:
            assert {c["causa"] for c in f["causas"]} <= opciones
        causas = {c["causa"] for f in client.get("/api/v1/composicion-causas").json()["franjas"] for c in f["causas"]}
        assert "Estado de Ebriedad o Droga" in causas and "Distracción del Conductor" in causas

    def test_la_ebriedad_pesa_mas_en_la_madrugada_que_en_la_tarde(self, client):
        franjas = {f["franja"]: {c["causa"]: c["proporcion"] for c in f["causas"]} for f in client.get("/api/v1/composicion-causas").json()["franjas"]}
        assert franjas["Madrugada (00-05)"]["Estado de Ebriedad o Droga"] > franjas["Tarde (12-17)"]["Estado de Ebriedad o Droga"]

    def test_la_suma_por_causa_de_cada_prediccion_es_su_total_esperado(self, client):
        r = client.post(
            "/api/v1/predicciones",
            json={"distritos": ["san salvador", "soyapango"], "fecha_inicio": "2024-06-15", "fecha_fin": "2024-06-16"},
        )
        celdas = r.json()["predicciones"]
        assert len(celdas) == 16
        for c in celdas:
            assert sum(x["esperado"] for x in c["por_causa"]) == pytest.approx(c["siniestros_esperados"], rel=1e-9)
            assert sum(x["proporcion"] for x in c["por_causa"]) == pytest.approx(1.0)
            esperados = [x["esperado"] for x in c["por_causa"]]
            assert esperados == sorted(esperados, reverse=True)

    def test_por_causa_usa_la_composicion_de_la_franja_de_la_celda(self, client):
        composicion = {f["franja"]: {c["causa"]: c["proporcion"] for c in f["causas"]} for f in client.get("/api/v1/composicion-causas").json()["franjas"]}
        r = client.post(
            "/api/v1/predicciones",
            json={"distritos": ["san salvador"], "fecha_inicio": "2024-06-15", "fecha_fin": "2024-06-15", "franjas": ["Noche (18-23)"]},
        )
        celda = r.json()["predicciones"][0]
        for x in celda["por_causa"]:
            assert x["proporcion"] == pytest.approx(composicion["Noche (18-23)"][x["causa"]])
