from fastapi.testclient import TestClient

from api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_forecast_returns_requested_horizon():
    with TestClient(app) as client:
        response = client.post(
            "/forecast", json={"store": 1, "item": 1, "horizon_days": 7}
        )
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 7
        assert all(p["predicted_sales"] >= 0 for p in predictions)


def test_invalid_store_rejected():
    with TestClient(app) as client:
        response = client.post("/forecast", json={"store": 99, "item": 1})
        assert response.status_code == 422
def test_response_has_request_id_header():
    with TestClient(app) as client:
        response = client.get("/health")
        assert "x-request-id" in response.headers
def test_forecast_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    body = {"store": 1, "item": 1, "horizon_days": 3}
    with TestClient(app) as client:
        assert client.post("/forecast", json=body).status_code == 401
        assert client.post("/forecast", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/forecast", json=body, headers={"X-API-Key": "test-key"}).status_code == 200
        assert client.get("/health").status_code == 200
