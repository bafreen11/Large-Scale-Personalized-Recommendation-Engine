"""Step 7: weekly retraining DAG (Apache Airflow 2.x). Copy or symlink this file into $AIRFLOW_HOME/dags/.

Each task is one pipeline step and runs only if the previous one succeeded, so a broken
feature build never ships a model. The last step publishes the new models to Triton.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/mnt/e/Large-Scale Personalized Recommendation Engine"
ENV = f'cd "{PROJECT}" && source ~/venvs/recsys/bin/activate && export PYTHONPATH=. && '

STEPS = [
    ("csv_to_parquet", "python -m src.data.csv_to_parquet"),
    ("time_split", "python -m src.data.split"),
    ("genome_pca", "python -m src.features.genome_pca"),
    ("spark_features", "bash scripts/run_phase2.sh --fg"),
    ("training_sample", "python -m src.training.sample"),
    ("two_tower", "python -m src.retrieval.two_tower"),
    ("faiss_index", "python -m src.index.build_faiss"),
    ("rankers", "python -m src.ranking.train_ranker"),
    ("evaluate", "python -m src.evaluation.evaluate"),
    ("export_triton", "python -m src.serving.export_triton"),
]

with DAG(
    dag_id="recsys_retrain",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["recsys"],
) as dag:
    tasks = [BashOperator(task_id=name, bash_command=ENV + cmd) for name, cmd in STEPS]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
