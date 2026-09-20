from fastapi.testclient import TestClient

from churn_mlops.api import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_predict_requires_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from churn_mlops.api import get_model

    get_model.cache_clear()
    response = TestClient(app).post(
        "/predict",
        json={
            "tenure_months": 6,
            "monthly_charges": 105,
            "support_tickets": 4,
            "contract_months": 1,
            "autopay": 0,
            "streaming_services": 2,
        },
    )
    assert response.status_code == 503

