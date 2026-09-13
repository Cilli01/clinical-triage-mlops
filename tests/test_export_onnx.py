import joblib
import numpy as np
import onnxruntime as ort
import pytest

from src.model.export_onnx import export_onnx
from src.model.train import build_pipeline


def _fit_tiny_pipeline():
    texts = [
        "cardiac stroke infarction heart failure urgent",
        "brain seizure neurological emergency stroke",
        "tumor neoplasm cancer malignant oncology",
        "gastric hepatic infection digestive cancer",
        "routine checkup general examination fever",
        "common cold general medical evaluation",
    ]
    labels = [
        "urgent",
        "urgent",
        "attention",
        "attention",
        "normal",
        "normal",
    ]
    pipeline = build_pipeline(max_features=200, n_estimators=15)
    pipeline.fit(texts, labels)
    return pipeline, texts


def test_export_onnx_creates_artifacts(tmp_path):
    pipeline, texts = _fit_tiny_pipeline()
    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    tfidf_path = tmp_path / "tfidf.joblib"
    joblib.dump(pipeline, model_path)

    export_onnx(model_path=model_path, onnx_path=onnx_path, tfidf_path=tfidf_path)

    assert onnx_path.exists()
    assert tfidf_path.exists()

    tfidf = joblib.load(tfidf_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    features = tfidf.transform([texts[0]]).astype(np.float32).toarray()
    labels, _probabilities = session.run(None, {session.get_inputs()[0].name: features})

    assert str(labels[0]) in {"normal", "attention", "urgent"}


def test_export_onnx_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_onnx(
            model_path=tmp_path / "missing.joblib",
            onnx_path=tmp_path / "out.onnx",
            tfidf_path=tmp_path / "tfidf.joblib",
        )
