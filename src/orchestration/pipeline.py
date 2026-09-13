"""Etapas reutilizáveis do pipeline de retreino."""

from typing import Any

from src.data.preprocess import preprocess_data
from src.model.export_onnx import export_onnx
from src.model.train import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    train_model,
)

RAW_TRAIN_PATH = PROJECT_ROOT / "data/raw/medical_tc_train.csv"
RAW_TEST_PATH = PROJECT_ROOT / "data/raw/medical_tc_test.csv"
PROCESSED_TRAIN_PATH = PROJECT_ROOT / "data/processed/medical_tc_train_processed.csv"
PROCESSED_TEST_PATH = PROJECT_ROOT / "data/processed/medical_tc_test_processed.csv"


def task_preprocess() -> dict[str, str]:
    """Gera os CSVs processados a partir dos dados brutos."""
    preprocess_data(RAW_TRAIN_PATH, PROCESSED_TRAIN_PATH)
    preprocess_data(RAW_TEST_PATH, PROCESSED_TEST_PATH)
    return {
        "train": str(PROCESSED_TRAIN_PATH),
        "test": str(PROCESSED_TEST_PATH),
    }


def task_train() -> dict[str, Any]:
    """Treina o classificador e persiste o artefato sklearn."""
    _, metrics = train_model(
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        model_path=DEFAULT_MODEL_PATH,
        metrics_path=DEFAULT_METRICS_PATH,
    )
    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
    }


def task_export_onnx() -> dict[str, str]:
    """Exporta o Random Forest treinado para ONNX Runtime."""
    onnx_path = export_onnx()
    return {"onnx_path": str(onnx_path)}


def run_retrain_pipeline() -> dict[str, Any]:
    """Executa preprocessamento, treino e exportação ONNX em sequência."""
    preprocess_result = task_preprocess()
    train_result = task_train()
    export_result = task_export_onnx()
    return {
        "preprocess": preprocess_result,
        "train": train_result,
        "export": export_result,
    }


def main() -> None:
    result = run_retrain_pipeline()
    print("Pipeline de retreino concluído.")
    print(f"Treino processado: {result['preprocess']['train']}")
    print(f"Modelo: {result['train']['model_path']}")
    print(f"F1 macro: {result['train']['f1_macro']:.4f}")
    print(f"ONNX: {result['export']['onnx_path']}")


if __name__ == "__main__":
    main()
