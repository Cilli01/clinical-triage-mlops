from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success():
    payload = {
        "condition_label": 4,
        "medical_abstract": "Patient presenting acute cardiac symptoms"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["urgency_prediction"] == "urgent"
    assert "confidence_score" in data
    assert isinstance(data["confidence_score"], float)
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0

def test_predict_empty_abstract():
    payload = {
        "condition_label": 4,
        "medical_abstract": ""
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "Medical abstract is required"}

def test_predict_invalid_payload():
    payload = {
        "condition_label": 4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    