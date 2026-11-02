"""Tests end-to-end de la API (httpx TestClient contra la app real, con los
artefactos versionados del repositorio)."""

from __future__ import annotations

import pytest

from app.infrastructure.geo.geojson_adapter import GeoJsonAdapter
from app.infrastructure.web.dependencias import get_geojson_adapter
from app.main import app


class TestSalud:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["estado"] == "ok"
        assert body["modelo_cargado"] is True
        assert body["prediccion_canaria_ok"] is True
        assert body["log_tasa_base"] == -12.936335060885538


class TestModeloInfo:
    def test_info_incluye_metricas_y_limitaciones(self, client):
        r = client.get("/api/v1/modelo/info")
        assert r.status_code == 200
        body = r.json()
        assert body["variables"] == [
            "es_lluviosa", "es_finde", "dia_semana", "periodo_agostino",
            "es_feriado", "prcp_mensual", "pct_urbano", "poblacion_baja",
        ]
        assert body["metricas"]["calibracion"] == 0.966979
        assert len(body["limitaciones"]) > 0


class TestDistritos:
    def test_lista_103_distritos(self, client):
        r = client.get("/api/v1/distritos")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 103
        assert len(body["distritos"]) == 103

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
        peaton = next(t for t in body["reparto_por_tipo"] if t["tipo"] == "Peaton")
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
            "Colision", "Atropello", "Caracteristicas especiales",
            "Choque contra objeto fijo", "Volcamiento",
        }
