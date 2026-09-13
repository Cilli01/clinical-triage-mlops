from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.model.train import build_pipeline
from src.orchestration import pipeline as orch


def test_task_preprocess(tmp_path, monkeypatch):
    raw_train = tmp_path / "raw_train.csv"
    raw_test = tmp_path / "raw_test.csv"
    processed_train = tmp_path / "train_processed.csv"
    processed_test = tmp_path / "test_processed.csv"

    frame = pd.DataFrame(
        {
            "condition_label": [1, 3, 5],
            "medical_abstract": [
                "Tumor neoplasm cancer case",
                "Stroke brain neurological case",
                "Routine general checkup case",
            ],
        }
    )
    frame.to_csv(raw_train, index=False)
    frame.to_csv(raw_test, index=False)

    monkeypatch.setattr(orch, "RAW_TRAIN_PATH", raw_train)
    monkeypatch.setattr(orch, "RAW_TEST_PATH", raw_test)
    monkeypatch.setattr(orch, "PROCESSED_TRAIN_PATH", processed_train)
    monkeypatch.setattr(orch, "PROCESSED_TEST_PATH", processed_test)

    result = orch.task_preprocess()

    assert Path(result["train"]).exists()
    assert Path(result["test"]).exists()
    assert "urgency" in pd.read_csv(processed_train).columns


def test_task_export_onnx(tmp_path):
    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    tfidf_path = tmp_path / "tfidf.joblib"

    texts = [
        "cardiac stroke infarction heart failure urgent case",
        "cardiac stroke infarction heart failure urgent patient",
        "tumor neoplasm cancer malignant oncology lesion",
        "tumor neoplasm cancer malignant oncology mass",
        "routine checkup general examination mild fever",
        "routine checkup general examination common cold",
    ]
    labels = [
        "urgent",
        "urgent",
        "attention",
        "attention",
        "normal",
        "normal",
    ]
    pipeline = build_pipeline(max_features=100, n_estimators=10)
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, model_path)

    from src.model import export_onnx as export_mod

    result_path = export_mod.export_onnx(
        model_path=model_path,
        onnx_path=onnx_path,
        tfidf_path=tfidf_path,
    )

    assert result_path == onnx_path
    assert onnx_path.exists()
    assert tfidf_path.exists()


def test_dag_file_defines_expected_flow():
    pytest.importorskip("airflow")

    dag_path = (
        Path(__file__).resolve().parents[1] / "dags" / "clinical_triage_retrain.py"
    )
    namespace = {"__file__": str(dag_path)}
    code = dag_path.read_text(encoding="utf-8")
    exec(compile(code, str(dag_path), "exec"), namespace)

    dag = namespace["dag"]
    assert dag.dag_id == "clinical_triage_retrain"
    assert set(dag.task_ids) == {
        "preprocess_data",
        "train_model",
        "export_onnx",
    }

    downstream = {t.task_id: {d.task_id for d in t.downstream_list} for t in dag.tasks}
    assert downstream["preprocess_data"] == {"train_model"}
    assert downstream["train_model"] == {"export_onnx"}
    assert downstream["export_onnx"] == set()
