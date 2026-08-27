import pandas as pd
import pytest

from src.data.preprocess import preprocess_data


def test_preprocess_data(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    input_data = pd.DataFrame(
        {
            "condition_label": [1, 3, 5],
            "medical_abstract": [
                "Patient with neoplasm.",
                "Patient with neurological symptoms.",
                "General medical examination.",
            ],
        }
    )

    input_data.to_csv(input_file, index=False)

    result = preprocess_data(input_file, output_file)

    assert "urgency" in result.columns
    assert result["urgency"].tolist() == ["attention", "urgent", "normal"]
    assert output_file.exists()


def test_missing_input_file(tmp_path):
    input_file = tmp_path / "does_not_exist.csv"
    output_file = tmp_path / "output.csv"

    with pytest.raises(FileNotFoundError):
        preprocess_data(input_file, output_file)


def test_missing_condition_label(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    input_data = pd.DataFrame(
        {
            "medical_abstract": [
                "Medical report without condition label.",
            ]
        }
    )

    input_data.to_csv(input_file, index=False)

    with pytest.raises(KeyError):
        preprocess_data(input_file, output_file)
