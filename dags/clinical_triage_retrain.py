"""DAG de retreino do classificador de triagem clínica."""

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.pipeline import (  # noqa: E402
    task_export_onnx,
    task_preprocess,
    task_train,
)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="clinical_triage_retrain",
    description="Preprocessa dados, retreina o NLP e exporta o modelo ONNX",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "triagem", "nlp"],
) as dag:
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
    )

    export = PythonOperator(
        task_id="export_onnx",
        python_callable=task_export_onnx,
    )

    preprocess >> train >> export
