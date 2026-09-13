"""Exporta o Random Forest treinado para ONNX Runtime."""

from pathlib import Path
from typing import Union

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from .train import DEFAULT_MODEL_PATH, PROJECT_ROOT

DEFAULT_ONNX_PATH = PROJECT_ROOT / "models/urgency_classifier.onnx"
DEFAULT_TFIDF_PATH = PROJECT_ROOT / "models/urgency_tfidf.joblib"


def export_onnx(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    onnx_path: Union[str, Path] = DEFAULT_ONNX_PATH,
    tfidf_path: Union[str, Path] = DEFAULT_TFIDF_PATH,
) -> Path:
    """Converte o classificador do pipeline sklearn para ONNX.

    O TF-IDF permanece em joblib (leve) e é reutilizado na inferência ONNX.
    A conversão do texto completo via StringNormalizer costuma falhar em
    ambientes sem locale en_US, então exportamos só o Random Forest.
    """
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {source}")

    pipeline: Pipeline = joblib.load(source)
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    n_features = len(tfidf.get_feature_names_out())
    onnx_model = convert_sklearn(
        classifier,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset=15,
        options={id(classifier): {"zipmap": False}},
    )

    out_onnx = Path(onnx_path)
    out_tfidf = Path(tfidf_path)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    out_onnx.write_bytes(onnx_model.SerializeToString())
    joblib.dump(tfidf, out_tfidf)

    return out_onnx


def main() -> None:
    onnx_path = export_onnx()
    print(f"ONNX salvo em: {onnx_path}")
    print(f"TF-IDF salvo em: {DEFAULT_TFIDF_PATH}")


if __name__ == "__main__":
    main()
