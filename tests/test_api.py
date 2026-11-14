import numpy as np
from fastapi.testclient import TestClient

from telecom_churn.api import create_app
from telecom_churn.scoring import score_customers


def test_api_predictions_equal_batch(model_dir, payload, latest, bundle):
    with TestClient(create_app(model_dir)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        assert response.headers["X-Request-ID"]
        expected = score_customers(latest.head(3), bundle)
        np.testing.assert_allclose(
            [row["risk_score"] for row in response.json()], expected["risk_score"]
        )
        assert (
            client.get("/api/v1/model").json()["model_version"]
            == bundle["metadata"]["model_version"]
        )


def test_authentication(model_dir, payload):
    with TestClient(create_app(model_dir, api_key="test-secret")) as client:
        assert client.post("/api/v1/predict", json=payload).status_code == 401
        assert client.get("/api/v1/model").status_code == 401
        assert (
            client.post(
                "/api/v1/predict", json=payload, headers={"X-API-Key": "test-secret"}
            ).status_code
            == 200
        )
        assert client.get("/health").status_code == 200


def test_unavailable_model(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_invalid_payloads(model_dir, payload):
    with TestClient(create_app(model_dir)) as client:
        assert client.post("/api/v1/predict", json={"customers": []}).status_code == 422
        duplicate = {"customers": payload["customers"] * 2}
        assert client.post("/api/v1/predict", json=duplicate).status_code == 422
        assert (
            client.post(
                "/api/v1/predict", json={"customers": payload["customers"] * 334}
            ).status_code
            == 422
        )
        payload["customers"][0]["churn"] = 1
        assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_production_requires_auth(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setenv("CHURN_ENV", "production")
    with pytest.raises(ValueError, match="CHURN_API_KEY"):
        create_app(tmp_path, api_key="")
