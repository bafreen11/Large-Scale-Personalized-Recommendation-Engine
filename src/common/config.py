import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "movie_lens_data"
PARQUET_DIR = ROOT / "data" / "parquet"        # raw CSVs converted to Parquet
PROCESSED_DIR = ROOT / "data" / "processed"    # cleaned splits + feature tables
MODELS_DIR = ROOT / "models"

# A rating >= this counts as a positive interaction ("click")
POSITIVE_RATING = 4.0

# Time split: train = before 90th percentile of timestamps, val = 90-95%, test = last 5%
TRAIN_QUANTILE = 0.90
VAL_QUANTILE = 0.95

MIN_USER_RATINGS = 5
MIN_ITEM_RATINGS = 5

SEED = 42

# ---- Steps 3-7: models, index, serving ----
SPLITS_DIR = ROOT / "data" / "splits"          # step 1 output (train / val / test / cold_start / movies_clean)
FEATURES_DIR = ROOT / "data" / "features"      # step 2 output
TRAINING_DIR = ROOT / "data" / "training"      # user sample used by the models (src.training.sample)
SAMPLE_PCT = int(os.environ.get("SAMPLE_PCT", "5"))   # % of users to train/evaluate on; 100 = everyone

SEQ_LEN = 50                 # must match src/features/config.py
N_GENRES = 19
PCA_K = 32

EMBED_DIM = 64               # two-tower output size (= FAISS vector size)
N_CANDIDATES = 200           # retrieved per user by the two-tower + FAISS
N_POPULAR_CANDIDATES = 50    # plus this many trending movies; the ranker re-scores all of them
TOP_K = 10

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
TRITON_URL = os.environ.get("TRITON_URL", "")  # e.g. localhost:8000; empty -> run models in-process
