"""Carregamento e inferência do classificador de urgência."""

import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "urgency_classifier.joblib"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "urgency_classifier.onnx"
DEFAULT_TFIDF_PATH = PROJECT_ROOT / "models" / "urgency_tfidf.joblib"

_pipeline: Optional[Pipeline] = None
_tfidf = None
_onnx_session: Optional[ort.InferenceSession] = None
_onnx_input_name: Optional[str] = None
_backend: Optional[str] = None


def get_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def get_onnx_path() -> Path:
    return Path(os.getenv("ONNX_PATH", str(DEFAULT_ONNX_PATH)))


def get_tfidf_path() -> Path:
    return Path(os.getenv("TFIDF_PATH", str(DEFAULT_TFIDF_PATH)))


def reset_pipeline() -> None:
    """Limpa o modelo em memória (útil principalmente nos testes)."""
    global _pipeline, _tfidf, _onnx_session, _onnx_input_name, _backend
    _pipeline = None
    _tfidf = None
    _onnx_session = None
    _onnx_input_name = None
    _backend = None


def resolve_backend() -> str:
    """Define o backend de inferência: onnx, sklearn ou auto."""
    configured = os.getenv("MODEL_BACKEND", "auto").strip().lower()
    if configured in {"onnx", "sklearn"}:
        return configured

    if get_onnx_path().exists() and get_tfidf_path().exists():
        return "onnx"
    return "sklearn"


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
    """Retorna o pipeline sklearn em cache."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


def get_onnx_runtime() -> Tuple[object, ort.InferenceSession, str]:
    """Carrega TF-IDF + sessão ONNX sob demanda."""
    global _tfidf, _onnx_session, _onnx_input_name

    if _onnx_session is None or _tfidf is None or _onnx_input_name is None:
        onnx_path = get_onnx_path()
        tfidf_path = get_tfidf_path()

        if not onnx_path.exists() or not tfidf_path.exists():
            raise FileNotFoundError(
                "Artefatos ONNX não encontrados. "
                "Execute: python -m src.model.export_onnx"
            )

        _tfidf = joblib.load(tfidf_path)
        _onnx_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        _onnx_input_name = _onnx_session.get_inputs()[0].name

    return _tfidf, _onnx_session, _onnx_input_name


def _predict_sklearn(medical_abstract: str) -> Tuple[str, float]:
    pipeline = get_pipeline()
    probabilities = pipeline.predict_proba([medical_abstract])[0]
    best_index = int(probabilities.argmax())
    urgency = str(pipeline.classes_[best_index])
    confidence = float(probabilities[best_index])
    return urgency, confidence


def _predict_onnx(medical_abstract: str) -> Tuple[str, float]:
    tfidf, session, input_name = get_onnx_runtime()
    features = tfidf.transform([medical_abstract]).astype(np.float32).toarray()
    labels, probabilities = session.run(None, {input_name: features})
    urgency = str(labels[0])
    confidence = float(np.max(probabilities[0]))
    return urgency, confidence


def predict_urgency(medical_abstract: str) -> Tuple[str, float]:
    """Classifica o laudo e devolve a classe prevista com a confiança."""
    global _backend
    if _backend is None:
        _backend = resolve_backend()

    if _backend == "onnx":
        try:
            return _predict_onnx(medical_abstract)
        except FileNotFoundError:
            # fallback se auto/onnx estiver inconsistente
            _backend = "sklearn"

    return _predict_sklearn(medical_abstract)
