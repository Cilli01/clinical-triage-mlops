"""Benchmark de latência: sklearn (baseline) vs ONNX Runtime."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import onnxruntime as ort

from .export_onnx import DEFAULT_ONNX_PATH, DEFAULT_TFIDF_PATH
from .train import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_PATH,
    PROJECT_ROOT,
    TEXT_COLUMN,
    _load_dataset,
)

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "models/latency_benchmark.json"


def _measure(
    predict_fn: Callable[[str], Any],
    texts: Sequence[str],
    warmup: int = 5,
    rounds: int = 30,
) -> dict[str, float]:
    for text in texts[:warmup]:
        predict_fn(text)

    total_predictions = rounds * len(texts)
    start = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            predict_fn(text)
    elapsed = time.perf_counter() - start

    return {
        "latency_ms": (elapsed / total_predictions) * 1000,
        "throughput_rps": total_predictions / elapsed,
        "n_texts": float(len(texts)),
        "n_rounds": float(rounds),
    }


def run_benchmark(
    model_path: Path = DEFAULT_MODEL_PATH,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    tfidf_path: Path = DEFAULT_TFIDF_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    sample_size: int = 50,
    output_path: Path = DEFAULT_BENCHMARK_PATH,
) -> dict[str, Any]:
    """Compara latência/vazão do pipeline sklearn com o classificador ONNX."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo sklearn não encontrado em: {model_path}")
    if not onnx_path.exists() or not tfidf_path.exists():
        raise FileNotFoundError(
            "Artefatos ONNX não encontrados. Execute: python -m src.model.export_onnx"
        )

    test_df = _load_dataset(test_path)
    texts = (
        test_df[TEXT_COLUMN]
        .sample(n=min(sample_size, len(test_df)), random_state=42)
        .tolist()
    )

    pipeline = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict_sklearn(text: str) -> str:
        return str(pipeline.predict([text])[0])

    def predict_onnx(text: str) -> str:
        features = tfidf.transform([text]).astype(np.float32).toarray()
        label = session.run(None, {input_name: features})[0][0]
        return str(label)

    sklearn_stats = _measure(predict_sklearn, texts)
    onnx_stats = _measure(predict_onnx, texts)

    f1_macro = None
    accuracy = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        f1_macro = metrics.get("f1_macro")
        accuracy = metrics.get("accuracy")

    result = {
        "sklearn": {
            "model": "TF-IDF + Random Forest",
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "latency_ms": round(sklearn_stats["latency_ms"], 3),
            "throughput_rps": round(sklearn_stats["throughput_rps"], 2),
        },
        "onnx": {
            "model": "TF-IDF + Random Forest (ONNX Runtime)",
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "latency_ms": round(onnx_stats["latency_ms"], 3),
            "throughput_rps": round(onnx_stats["throughput_rps"], 2),
        },
        "sample_size": len(texts),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    return result


def main() -> None:
    result = run_benchmark()
    sk = result["sklearn"]
    ox = result["onnx"]

    print("Benchmark concluído.")
    print(
        f"Sklearn  | F1={sk['f1_macro']:.4f} | "
        f"{sk['latency_ms']:.3f} ms | {sk['throughput_rps']:.2f} req/s"
    )
    print(
        f"ONNX     | F1={ox['f1_macro']:.4f} | "
        f"{ox['latency_ms']:.3f} ms | {ox['throughput_rps']:.2f} req/s"
    )
    print(f"Resultado salvo em: {DEFAULT_BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
