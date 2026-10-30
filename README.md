# Large-Scale Personalized Recommendation Engine

An end-to-end movie recommender on **MovieLens 25M** (25M ratings, 162K users, 62K movies). It runs on a laptop
(4 cores, 12 GB RAM, no GPU) under WSL Ubuntu:

```
ratings ──► time split ──► Spark features ──► two-tower (TFRS) ──► FAISS ──► DCN / Wide&Deep ──► top-10
                                                                             ▲                    │
                     new users: popularity / content-based fallback ─────────┘        Redis cache ┘
                     serving: FastAPI (+ Triton) · retraining: Airflow · A/B test: DCN vs Wide&Deep
```

## Project status

| # | Step | Status | Code |
|---|------|--------|------|
| 1 | Clean MovieLens 25M, split **by time** (90% / 5% / 5%) | ✅ | `src/data/` |
| 2 | Spark interaction tables + features (user history, item popularity, genres, genome tags) | ✅ full data | `src/features/` |
| 3 | Two-tower retrieval model in TensorFlow Recommenders | ✅ trained on a sample | `src/retrieval/` |
| 4 | FAISS index of item embeddings | ✅ | `src/index/` |
| 5 | Deep ranking: DCN-v2 and Wide & Deep | ✅ trained on a sample | `src/ranking/` |
| 6 | Cold start (popularity + content-based) and Redis cache | ✅ | `src/serving/` |
| 7 | Triton serving, Airflow retraining, A/B test (NDCG, Recall@K, CTR) | ✅ code + config; A/B test run as a replay | `src/serving/`, `src/evaluation/`, `triton/`, `airflow/` |

Steps 3–7 were trained and evaluated on **8% of users** (12K users, 1.8M ratings) so each model trains in
minutes on CPU. `SAMPLE_PCT=100` runs everyone, given enough RAM.

## Results (8% user sample, val period)

Offline evaluation (`reports/eval_report_val.json`): 432 warm users, top-10. Movies a user already rated are never recommended.

| Method | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity (last 30 days) | **0.0525** | **0.0825** |
| Two-tower + FAISS | 0.0253 | 0.0467 |
| Two-tower + trending → **DCN** | 0.0411 | 0.0597 |
| Two-tower + trending → Wide & Deep | 0.0293 | 0.0577 |

| Component | Result |
|---|---|
| Ranker AUC on the val period | DCN **0.776**, Wide & Deep 0.768 |
| Candidate recall (200 two-tower + 50 trending) | 36.8% of relevant movies reach the ranker |
| FAISS IVF vs exact search | 90.8% of the exact top-200 found, **0.105 ms vs 0.77 ms** per query (26K items) |
| Cold start (995 new users, top-10) | popularity NDCG 0.298 · content fallback 0.257 |
| Serving latency | 7.9 ms uncached (tower + FAISS + ranker), 0.002 ms from cache |

A/B test (`reports/ab_test_report.json`): a replay of the test period, with users split 50/50 by a stable hash.

| Variant | Users | Impressions | Clicks | CTR | NDCG@10 | Recall@10 |
|---|---|---|---|---|---|---|
| DCN | 143 | 1,430 | 76 | **5.3%** | **0.066** | **0.035** |
| Wide & Deep | 146 | 1,460 | 64 | 4.4% | 0.052 | 0.031 |

z = 1.17, p = 0.24: DCN is ahead on every metric but **not statistically significant** at this sample size.

**Honest reading.** With a global time split, the next period is dominated by trending releases, and recent popularity is a
strong baseline. The personalised models don't beat it yet on this small sample. DCN on top of mixed candidates
gets closest. The most promising next steps are:

- train on more users
- add sampled negatives for the ranker; it currently learns only from rated movies
- add recency features to the user tower

## Repository layout

```
src/
  common/       config.py (paths, model sizes) · spark.py (local Spark session for WSL)
  data/         step 1  csv_to_parquet.py · split.py (time split, min-5 filter, warm/cold split)
  features/     step 2  run_phase2.py · user/item/interaction features · genome_pca.py · config.py
  training/     sample.py (user sample for models) · data.py (feature contract, numpy loaders)
  retrieval/    step 3  two_tower.py
  index/        step 4  build_faiss.py
  ranking/      step 5  models.py (DCN, Wide&Deep) · train_ranker.py
  serving/      step 6/7  recommender.py · cold_start.py · cache.py · models.py (local/Triton) · app.py · export_triton.py
  evaluation/   metrics.py · evaluate.py · ab_test.py
scripts/        run_phase2.sh (step 2) · run_pipeline.sh (steps 3-7)
triton/         model_repository/*/config.pbtxt
airflow/dags/   recsys_retrain.py (weekly retraining DAG)
reports/        JSON results of the last pipeline run
docker-compose.yml   Redis + Triton
```

## How to run

Run everything inside WSL Ubuntu from the project root, with Python 3.11 and Java 17+.

```bash
python3.11 -m venv ~/venvs/recsys && source ~/venvs/recsys/bin/activate
pip install -r requirements.txt
# download ml-25m from https://grouplens.org/datasets/movielens/25m/ and unzip into movie_lens_data/

export PYTHONPATH=.
python -m src.data.csv_to_parquet        # step 1
python -m src.data.split
python -m src.features.genome_pca        # step 2 (once), then the Spark job:
scripts/run_phase2.sh                    #   runs in the background; tail -f ~/phase2_logs/latest.log
SAMPLE_PCT=8 scripts/run_pipeline.sh     # steps 3-7; tail -f ~/recsys_logs/latest.log
```

The scripts run as systemd user units, so a closed terminal or a VS Code disconnect doesn't kill a long Spark or training job.

### Serving

```bash
uvicorn src.serving.app:app --port 8080
curl "localhost:8080/recommend/39?k=10"                          # known user -> two-tower + ranker (A/B variant)
curl "localhost:8080/recommend/999999?k=10&liked=260,2571"       # new user -> content-based fallback
curl -X POST localhost:8080/feedback -H 'content-type: application/json' \
     -d '{"user_id": 39, "movie_id": 50, "variant": "dcn"}'      # log a click
curl localhost:8080/ab/report                                    # live CTR per variant + z-test
```

Optional infrastructure: the code falls back to an in-memory cache and in-process models when these aren't running.

```bash
docker compose up -d redis                                        # Redis cache
python -m src.serving.export_triton && docker compose up -d triton
TRITON_URL=localhost:8000 uvicorn src.serving.app:app --port 8080 # models served by Triton
```

### Retraining

`airflow/dags/recsys_retrain.py` chains every step weekly (`@weekly`, one retry, no catch-up). Copy it into
`$AIRFLOW_HOME/dags/`. Airflow is best installed in its own environment.

## Design notes

- **No leakage:** all features are built from the training period only. Training rows use *as-of-time* features
  (a user's history strictly before each rating), and `run_phase2.py` asserts this.
- **Two-tower:** shared movie embedding for the history and candidate towers, in-batch softmax with logQ correction
  (removing it made retrieval much worse) and accidental-hit removal.
- **One feature contract** (`src/training/data.py`) is shared by training, evaluation, the API and Triton configs.
- **Laptop-friendly Spark:** `local[3]`, a 4 GB driver, and scratch space on disk. WSL's `/tmp` lives in RAM.
