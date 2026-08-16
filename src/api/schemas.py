from pydantic import BaseModel, Field


class MedicalReport(BaseModel):
    condition_label: int = Field(
        ..., description="Condição clínica detectada no relatório"
    )
    medical_abstract: str = Field(..., description="Resumo médico do relatório")


class PredictResponses(BaseModel):
    urgency_prediction: str = Field(..., description="Nível de urgência detectado")
    confidence_score: float = Field(
        ..., description="Pontuação de confiança da predição"
    )
    latency_ms: float = Field(..., description="Latência do modelo em milissegundos")
