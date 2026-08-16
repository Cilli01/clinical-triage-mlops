import time

from fastapi import FastAPI, HTTPException

from .schemas import MedicalReport, PredictResponses

app = FastAPI(
    title="Clinical Triage MLOps",
    description="Sistema de triagem clínica baseado em aprendizado de máquina.",
    version="0.0.1",
)


@app.get("/health")
def health_check():
    """Retorna o status da API para monitoramento de infraestrutura."""
    return {"status": "ok"}


@app.post("/predict")
def predict(report: MedicalReport):
    """Retorna a predição de urgência para um relatório médico.

    Args:
        report: Relatório médico contendo o resumo e a condição clínica detectada

    Returns:
        Dicionário contendo a predição de urgência, pontuação de confiança e latência
    """
    start_time = time.perf_counter()

    if not report.medical_abstract.strip():
        raise HTTPException(status_code=400, detail="Medical abstract is required")

    medical_abstract_lower = report.medical_abstract.strip().lower()

    # mock
    if any(
        word in medical_abstract_lower
        for word in ["heart", "cardiac", "stroke", "infarction", "apnea", "brain"]
    ):
        urgency = "urgent"
        confidence = 0.95
    elif any(
        word in medical_abstract_lower
        for word in ["stomach", "gastric", "tumor", "cancer", "hepatic", "infection"]
    ):
        urgency = "attention"
        confidence = 0.82
    else:
        urgency = "normal"
        confidence = 0.70

    latency_ms = float((time.perf_counter() - start_time) * 1000)

    return PredictResponses(
        urgency_prediction=urgency, confidence_score=confidence, latency_ms=latency_ms
    )
