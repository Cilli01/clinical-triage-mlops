import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .predictor import predict_urgency
from .schemas import MedicalReport, PredictResponses

app = FastAPI(
    title="Clinical Triage MLOps",
    description="Sistema de triagem clínica baseado em aprendizado de máquina.",
    version="0.0.1",
)


# =====================================================================
# Métricas Prometheus (expostas em /metrics para scrape)
# =====================================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total de requisições HTTP recebidas na API",
    ["endpoint", "method", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latência das requisições HTTP em segundos",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total de predições realizadas por classe de urgência",
    ["urgency_class"],
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total de erros ocorridos na API",
    ["endpoint", "error_type"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Mede latência e conta requisições em todos os endpoints automaticamente."""
    start_time = time.perf_counter()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        ERROR_COUNT.labels(endpoint=endpoint, error_type="exception").inc()
        raise
    finally:
        latency = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        try:
            status_label = str(status_code)
        except NameError:
            status_label = "500"
        REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, http_status=status_label
        ).inc()

    return response


@app.get("/health")
def health_check():
    """Retorna o status da API para monitoramento de infraestrutura."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Expõe métricas no formato Prometheus para scrape."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(report: MedicalReport):
    """Retorna a predição de urgência para um relatório médico.

    Args:
        report: Relatório médico contendo o resumo e a condição clínica detectada

    Returns:
        Dicionário contendo a predição de urgência, pontuação de confiança e latência
    """
    start_time = time.perf_counter()

    abstract = report.medical_abstract.strip()
    if not abstract:
        ERROR_COUNT.labels(endpoint="/predict", error_type="empty_abstract").inc()
        raise HTTPException(status_code=400, detail="Medical abstract is required")

    try:
        urgency, confidence = predict_urgency(abstract)
    except FileNotFoundError as exc:
        ERROR_COUNT.labels(endpoint="/predict", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    PREDICTION_COUNT.labels(urgency_class=urgency).inc()

    latency_ms = float((time.perf_counter() - start_time) * 1000)

    return PredictResponses(
        urgency_prediction=urgency, confidence_score=confidence, latency_ms=latency_ms
    )
