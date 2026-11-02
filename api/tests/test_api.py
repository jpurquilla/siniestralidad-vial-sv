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
