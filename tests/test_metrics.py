from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _metric_value(metrics_text: str, metric_name: str) -> float:
    """Soma o valor de uma métrica (ou de uma série com labels específicos)."""
    total = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue

        name_part, _, value_part = line.rpartition(" ")
        base_name = name_part.split("{", 1)[0]

        if base_name != metric_name.split("{", 1)[0]:
            continue
        if "{" in metric_name and not name_part.startswith(metric_name):
            continue

        total += float(value_part)

    return total


def test_metrics_endpoint_exposes_prometheus_format():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "api_requests_total" in response.text
    assert "api_request_latency_seconds" in response.text
    assert "predictions_total" in response.text
    assert "api_errors_total" in response.text


def test_request_count_increments_after_health_call():
    before = _metric_value(client.get("/metrics").text, "api_requests_total")

    client.get("/health")

    after = _metric_value(client.get("/metrics").text, "api_requests_total")

    assert after == before + 2  # a própria chamada a /metrics também é contada


def test_prediction_count_increments_by_urgency_class():
    metric = 'predictions_total{urgency_class="urgent"}'
    before = _metric_value(client.get("/metrics").text, metric)

    client.post(
        "/predict",
        json={
            "condition_label": 4,
            "medical_abstract": "Patient with acute cardiac arrest",
        },
    )

    after = _metric_value(client.get("/metrics").text, metric)

    assert after == before + 1


def test_error_count_increments_on_empty_abstract():
    before = _metric_value(client.get("/metrics").text, "api_errors_total")

    client.post("/predict", json={"condition_label": 4, "medical_abstract": "   "})

    after = _metric_value(client.get("/metrics").text, "api_errors_total")

    assert after == before + 1
