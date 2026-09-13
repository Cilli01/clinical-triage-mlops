"""Carregamento e inferência do classificador de urgência."""

import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "urgency_classifier.joblib"
)

_pipeline: Optional[Pipeline] = None


def get_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def reset_pipeline() -> None:
    """Limpa o modelo em memória (útil principalmente nos testes)."""
    global _pipeline
    _pipeline = None


def load_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """Carrega o pipeline treinado a partir do disco."""
    path = model_path or get_model_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {path}. "
            "Execute o treino com: python -m src.model.train"
        )

    return joblib.load(path)


def get_pipeline() -> Pipeline:
    """Retorna o pipeline em cache, carregando na primeira chamada."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


def predict_urgency(medical_abstract: str) -> Tuple[str, float]:
    """Classifica o laudo e devolve a classe prevista com a confiança."""
    pipeline = get_pipeline()
    probabilities = pipeline.predict_proba([medical_abstract])[0]
    best_index = int(probabilities.argmax())
    urgency = str(pipeline.classes_[best_index])
    confidence = float(probabilities[best_index])
    return urgency, confidence
