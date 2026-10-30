# Large-Scale Personalized Recommendation Engine

MovieLens-25M recommender: time split -> Spark features -> two-tower (TFRS) -> FAISS -> DCN / Wide&Deep
ranking -> cold-start + Redis -> FastAPI / Triton / Airflow / A/B test. Runs inside **WSL Ubuntu**.
See README.md for the step-by-step layout and latest results.

## Environment
- Project path in Ubuntu: `/mnt/e/Large-Scale Personalized Recommendation Engine` (Windows E: drive, slow 9p filesystem)
- Python venv: `source ~/venvs/recsys/bin/activate` (Python 3.11, pyspark 4.0.1, TF 2.15, Java 21). Packages are
  managed with `uv` (`uv pip install --python ~/venvs/recsys/bin/python ...`); the venv has no pip.
- Run everything from the project root with `PYTHONPATH=.` (modules are `python -m src.<pkg>.<module>`).
- The Windows PATH leaks into Ubuntu and contains a Windows Spark 3.5.5 (`/mnt/c/spark/...`); always activate the venv first.
- Machine: 12 GB RAM, 4 cores, no GPU. WSL limit is 7 GB (`C:\Users\Mohammed Irfan\.wslconfig`).
  Spark: `--driver-memory 4g`, `local[3]`. Model training above ~8% of users risks OOM.
- `/tmp` is tmpfs (RAM). Never let Spark spill there; use `spark.local.dir=~/spark-tmp` (src/common/spark.py does).
- Long jobs: start them through the scripts (systemd user units) so a VS Code/WSL disconnect doesn't kill them.

## Pipeline
| step | command | output |
|---|---|---|
| 1 | `python -m src.data.csv_to_parquet && python -m src.data.split` | data/parquet, data/splits |
| 2 | `python -m src.features.genome_pca` (once), then `scripts/run_phase2.sh` | data/features |
| 3-7 | `SAMPLE_PCT=8 scripts/run_pipeline.sh` | data/training, models/, reports/ |

- Step 2 log: `~/phase2_logs/latest.log`; steps 3-7 log: `~/recsys_logs/latest.log`.
- Step 2 config: `src/features/config.py`. Steps 3-7 config: `src/common/config.py`.
- Feature contract shared by all models / Triton: `src/training/data.py` docstring. Changing it means
  updating `triton/model_repository/*/config.pbtxt` dims too.

## Debugging
- Real problems: `grep -E 'ERROR|Exception|Traceback|OutOfMemory' <log>`
- Job vanished with no error -> WSL was shut down from Windows (`journalctl --list-boots`), not a code bug.
- OOM -> lower SAMPLE_PCT or Spark parallelism before raising memory; WSL only has 7 GB.
- Known harmless noise: TF "Unable to register cuDNN/cuFFT/cuBLAS factory", Keras `get_config` warnings on export.
