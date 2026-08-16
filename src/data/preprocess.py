from pathlib import Path
from typing import Union

import pandas as pd


def preprocess_data(
    input_path: Union[str, Path], output_path: Union[str, Path]
) -> pd.DataFrame:
    """Carrega os laudos médicos brutos, mapeia as 5 especialidades originais
    para 3 níveis de urgência clínica e salva o resultado pré-processado.

    Args:
        input_path: Caminho para o arquivo CSV de entrada (dados brutos).
        output_path: Caminho onde o arquivo CSV processado será salvo.

    Returns:
        pd.DataFrame: DataFrame processado contendo a nova coluna 'urgency'.
    """

    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado em: {in_path}")

    df = pd.read_csv(in_path)

    # Mapeamento clínico acordado para triagem hospitalar
    urgency_mapping = {
        1: "attention",  # neoplasms
        2: "attention",  # digestive
        3: "urgent",  # nervous
        4: "urgent",  # cardiovascular
        5: "normal",  # general
    }

    if "condition_label" not in df.columns:
        raise KeyError(
            "A coluna obrigatória 'condition_label' não foi encontrada no dataset."
        )

    df_formatted = df.copy()
    df_formatted["urgency"] = df_formatted["condition_label"].map(urgency_mapping)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_formatted.to_csv(out_path, index=False)

    return df_formatted
