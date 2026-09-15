"""Tests end-to-end de la API (httpx TestClient contra la app real, con los
artefactos versionados del repositorio)."""

from __future__ import annotations


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

    def test_geojson_devuelve_503_si_no_existe(self, client):
        r = client.get("/api/v1/distritos/geojson")
        assert r.status_code == 503
        assert "GeoJSON" in r.json()["detail"] or "geojson" in r.json()["detail"].lower()


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
