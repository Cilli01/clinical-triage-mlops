import json

import pandas as pd
import pytest

from src.model.train import build_pipeline, train_model


def _write_tiny_dataset(path, labels):
    rows = {
        "condition_label": [1, 3, 5] * 4,
        "medical_abstract": [
            "Tumor neoplasm cancer malignant lesion oncology chemotherapy.",
            "Stroke brain seizure neurological infarction cerebral palsy.",
            "Routine checkup general examination mild fever common cold.",
        ]
        * 4,
        "urgency": labels * 4,
    }
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_pipeline_steps():
    pipeline = build_pipeline(max_features=100, n_estimators=10)
    assert list(pipeline.named_steps) == ["tfidf", "clf"]


def test_train_model_persists_artifacts(tmp_path):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    model_file = tmp_path / "models" / "urgency_classifier.joblib"
    metrics_file = tmp_path / "models" / "metrics.json"

    labels = ["attention", "urgent", "normal"]
    _write_tiny_dataset(train_file, labels)
    _write_tiny_dataset(test_file, labels)

    pipeline, metrics = train_model(
        train_path=train_file,
        test_path=test_file,
        model_path=model_file,
        metrics_path=metrics_file,
    )

    assert model_file.exists()
    assert metrics_file.exists()
    assert "accuracy" in metrics
    assert "f1_macro" in metrics

    saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved_metrics["model"] == "TF-IDF + Random Forest"

    prediction = pipeline.predict(
        ["Stroke brain seizure neurological infarction cerebral."]
    )[0]
    assert prediction in {"attention", "urgent", "normal"}


def test_train_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_model(
            train_path=tmp_path / "missing.csv",
            test_path=tmp_path / "also_missing.csv",
            model_path=tmp_path / "model.joblib",
            metrics_path=tmp_path / "metrics.json",
        )
