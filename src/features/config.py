"""Phase 2 config: the only file you should need to edit to match your Phase 1 output."""
import os

# ---- Inputs (adjust to what data/split.py wrote) ----
TRAIN_PATH   = "data/splits/train"
MOVIES_PATH  = "data/splits/movies_clean"
GENOME_PATH  = "data/parquet/genome_scores"

# ---- Outputs ----
# PHASE2_OUT_DIR lets run_phase2.sh write to the Linux disk (fast) instead of /mnt/e (slow 9p), then copy back
OUT_DIR = os.environ.get("PHASE2_OUT_DIR", "data/features")
USER_FEATURES_OUT  = f"{OUT_DIR}/user_features.parquet"
GENOME_PCA_OUT = "data/features/genome_pca.parquet"   # input, written by precompute_genome_pca.py
ITEM_FEATURES_OUT  = f"{OUT_DIR}/item_features.parquet"
TRAIN_INTERACTIONS_OUT = f"{OUT_DIR}/train_interactions.parquet"

# ---- Column names in the train table ----
USER, ITEM, RATING, TS = "userId", "movieId", "rating", "timestamp"

# ---- Modelling choices ----
POSITIVE_THRESHOLD = 4.0     # rating >= 4.0 -> label 1, else 0
SEQ_LEN   = 50               # last N movieIds per user
PCA_K     = 32               # genome PCA dimensions
N_GENOME_TAGS = 1128         # MovieLens genome has 1128 tags
DAY = 86400                  # seconds
CUTOFF_TS = None             # None -> max(train timestamp) + 1. Or put the cutoff Phase 1 used.

# Fixed genre list => fixed vector positions across all tables (order matters, never change later)
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
