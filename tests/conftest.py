import joblib
import pytest

from src.api.predictor import reset_pipeline
from src.model.train import build_pipeline


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """Gera um modelo pequeno para os testes da API, sem depender do artefato real."""
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "urgency_classifier.joblib"

    texts = [
        "Patient presenting acute cardiac symptoms stroke infarction",
        "Patient with acute cardiac arrest heart failure",
        "Stroke brain neurological emergency infarction",
        "Tumor neoplasm cancer malignant oncology lesion",
        "Gastric stomach hepatic infection digestive tumor",
        "Cancer chemotherapy neoplasm malignant mass",
        "Routine checkup general examination mild fever",
        "Common cold general medical evaluation checkup",
        "Annual physical exam general wellness visit",
    ]
    labels = [
        "urgent",
        "urgent",
        "urgent",
        "attention",
        "attention",
        "attention",
        "normal",
        "normal",
        "normal",
    ]

    pipeline = build_pipeline(max_features=500, n_estimators=20)
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture(autouse=True)
def _configure_test_model(trained_model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_path))
    monkeypatch.setenv("MODEL_BACKEND", "sklearn")
    reset_pipeline()
    yield
    reset_pipeline()
