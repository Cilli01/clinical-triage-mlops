"""Treino do classificador de urgência (TF-IDF + Random Forest)."""

import json
from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data/processed/medical_tc_train_processed.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data/processed/medical_tc_test_processed.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/urgency_classifier.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models/metrics.json"

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "urgency"


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {path}")

    df = pd.read_csv(path)

    missing = [col for col in (TEXT_COLUMN, TARGET_COLUMN) if col not in df.columns]
    if missing:
        raise KeyError(
            f"Colunas obrigatórias ausentes no dataset: {', '.join(missing)}"
        )

    df = df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""]

    return df


def build_pipeline(
    max_features: int = 5000,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Pipeline:
    """Monta o pipeline de vetorização TF-IDF + Random Forest."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_model(
    pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series
) -> dict[str, Any]:
    """Calcula métricas de classificação no conjunto de teste."""
    y_pred = pipeline.predict(X_test)

    return {
        "model": "TF-IDF + Random Forest",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "n_train_features": int(pipeline.named_steps["tfidf"].max_features or 0),
        "n_estimators": int(pipeline.named_steps["clf"].n_estimators),
    }


def train_model(
    train_path: Union[str, Path] = DEFAULT_TRAIN_PATH,
    test_path: Union[str, Path] = DEFAULT_TEST_PATH,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    metrics_path: Union[str, Path] = DEFAULT_METRICS_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Treina o modelo, avalia no teste e persiste artefato + métricas.

    Returns:
        Tupla com o pipeline treinado e o dicionário de métricas.
    """
    train_df = _load_dataset(Path(train_path))
    test_df = _load_dataset(Path(test_path))

    X_train = train_df[TEXT_COLUMN]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[TEXT_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    metrics["n_train_samples"] = int(len(train_df))
    metrics["n_test_samples"] = int(len(test_df))

    model_out = Path(model_path)
    metrics_out = Path(metrics_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_out)

    with metrics_out.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    return pipeline, metrics


def main() -> None:
    _, metrics = train_model()

    print("Treino concluído.")
    print(f"Modelo salvo em: {DEFAULT_MODEL_PATH}")
    print(f"Métricas salvas em: {DEFAULT_METRICS_PATH}")
    print(f"Acurácia: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
