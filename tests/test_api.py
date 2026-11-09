from fastapi.testclient import TestClient

from api.main import app, state


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
        assert (
            client.post(
                "/forecast", json=body, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/forecast", json=body, headers={"X-API-Key": "test-key"}
            ).status_code
            == 200
        )
        assert client.get("/health").status_code == 200


def test_unknown_store_item_returns_404():
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={"store": 2, "item": 1, "horizon_days": 7},
        )
        assert response.status_code == 404


def test_invalid_item_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={"store": 1, "item": 99, "horizon_days": 7},
        )
        assert response.status_code == 422


def test_invalid_horizon_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={"store": 1, "item": 1, "horizon_days": 91},
        )
        assert response.status_code == 422


def test_forecast_response_structure():
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={"store": 1, "item": 1, "horizon_days": 3},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["store"] == 1
        assert data["item"] == 1
        assert data["horizon_days"] == 3
        assert len(data["predictions"]) == 3

        # sourcery skip: no-loop-in-tests
        # sourcery skip: no-loop-in-tests
        for prediction in data["predictions"]:
            assert "date" in prediction
            assert "predicted_sales" in prediction


def test_forecast_unexpected_error_returns_safe_500(monkeypatch):
    def failing_forecast(store, item, horizon_days):
        raise RuntimeError("Sensitive internal error")

    with TestClient(app) as client:
        monkeypatch.setattr(
            state["forecaster"],
            "forecast",
            failing_forecast,
        )

        response = client.post(
            "/forecast",
            json={"store": 1, "item": 1, "horizon_days": 7},
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Forecast generation failed. Please try again later."
        }
        assert "Sensitive internal error" not in response.text


def test_middleware_handles_unexpected_exception():
    @app.get("/test-middleware-error")
    def raise_unexpected_error():
        raise RuntimeError("Internal test exception")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-middleware-error")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "x-request-id" in response.headers
    assert "Internal test exception" not in response.text


def test_metrics_returns_503_when_key_not_configured(monkeypatch):
    monkeypatch.delenv("METRICS_API_KEY", raising=False)

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 503


def test_metrics_rejects_missing_or_invalid_bearer_token(monkeypatch):
    monkeypatch.setenv("METRICS_API_KEY", "metrics-test-key")

    with TestClient(app) as client:
        missing = client.get("/metrics")
        invalid = client.get(
            "/metrics",
            headers={"Authorization": "Bearer wrong-key"},
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_metrics_accepts_valid_bearer_token(monkeypatch):
    monkeypatch.setenv("METRICS_API_KEY", "metrics-test-key")

    with TestClient(app) as client:
        response = client.get(
            "/metrics",
            headers={"Authorization": "Bearer metrics-test-key"},
        )

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert b"smartdemand_model_loaded" in response.content
