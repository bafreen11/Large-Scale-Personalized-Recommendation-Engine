"""Turn the data/training parquet tables into numpy feature dicts shared by every model.

Feature contract (same names in training, evaluation, serving and Triton):
    user_seq   int32   (n, SEQ_LEN)  last N item indices, oldest -> newest, right-aligned, 0 = padding
    user_dense float32 (n, 3 + N_GENRES)  log #ratings, avg rating, log #active days, genre affinity
    item_idx   int32   (n, 1)        item index into the catalogue (0 = unknown)
    item_dense float32 (n, ITEM_DENSE)  year, genres, log popularity (all/30d/90d), avg rating, genome PCA, has_genome
"""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.common.config import N_GENRES, PCA_K, SEQ_LEN, TRAINING_DIR

USER_DENSE = 3 + N_GENRES
ITEM_DENSE = 1 + N_GENRES + 4 + PCA_K + 1


def _scalar(table, name, fill=0):
    return pc.fill_null(table.column(name), fill).to_numpy().astype(np.float32)


def _lists(table, name, width, dtype=np.float32):
    """list column -> (n, width) matrix, right-aligned (for user_seq: newest item last), zero-padded."""
    col = table.column(name).combine_chunks()
    offsets = col.offsets.to_numpy()
    offsets = offsets - offsets[0]
    values = col.flatten().to_numpy(zero_copy_only=False)
    full = np.diff(offsets)
    lens = np.minimum(full, width)
    out = np.zeros((len(lens), width), dtype)
    rows = np.repeat(np.arange(len(lens)), lens)
    k = np.arange(len(rows)) - np.repeat(np.cumsum(lens) - lens, lens)   # 0..len-1 within each list
    src = np.repeat(offsets[:-1] + full - lens, lens) + k                 # the LAST `lens` values
    out[rows, width - lens[rows] + k] = values[src]
    return out


class Catalog:
    """All movies. movieId <-> item index (1..N, 0 is padding/unknown) and the item feature matrix."""

    def __init__(self, path=TRAINING_DIR / "items.parquet"):
        t = pq.read_table(path)
        order = np.argsort(t.column("movieId").to_numpy())
        t = t.take(order)
        self.movie_ids = t.column("movieId").to_numpy()
        self.titles = np.array(pc.fill_null(t.column("title"), "").to_pylist(), dtype=object)
        self.item_idx = np.arange(1, len(self.movie_ids) + 1, dtype=np.int32)
        self.pop_all = _scalar(t, "item_pop_all")
        self.pop_30d = _scalar(t, "item_pop_30d")
        self.dense = item_dense(t)
        self.genres = _lists(t, "item_genres", N_GENRES)
        self.genome = _lists(t, "item_genome_pca", PCA_K)
        self.has_genome = _scalar(t, "has_genome")

    def __len__(self):
        return len(self.movie_ids)

    def to_idx(self, movie_ids):
        movie_ids = np.asarray(movie_ids)
        pos = np.clip(np.searchsorted(self.movie_ids, movie_ids), 0, len(self.movie_ids) - 1)
        return np.where(self.movie_ids[pos] == movie_ids, pos + 1, 0).astype(np.int32)

    def features(self, idx):
        """Item tower / ranker inputs for item indices (1-based)."""
        idx = np.asarray(idx, np.int32)
        return {"item_idx": idx.reshape(-1, 1), "item_dense": self.dense[idx - 1]}


def item_dense(t):
    year = _scalar(t, "item_year")
    year = np.where(year > 0, (year - 1990.0) / 30.0, 0.0)[:, None]
    pops = np.log1p(np.stack([_scalar(t, c) for c in ["item_pop_all", "item_pop_30d", "item_pop_90d"]], 1)) / 10.0
    avg = (_scalar(t, "item_avg_rating", 3.5) / 5.0)[:, None]
    return np.hstack([year, _lists(t, "item_genres", N_GENRES), pops, avg,
                      _lists(t, "item_genome_pca", PCA_K) / 5.0, _scalar(t, "has_genome")[:, None]]).astype(np.float32)


def user_dense(t):
    return np.hstack([np.log1p(_scalar(t, "user_n_ratings"))[:, None] / 10.0,
                      (_scalar(t, "user_avg_rating", 3.5) / 5.0)[:, None],
                      np.log1p(_scalar(t, "user_days_active"))[:, None] / 10.0,
                      _lists(t, "user_genre_affinity", N_GENRES)]).astype(np.float32)


def user_seq(t, catalog):
    seq = _lists(t, "user_seq", SEQ_LEN, np.int64)
    return np.where(seq > 0, catalog.to_idx(seq), 0).astype(np.int32)


def load_interactions(catalog, path=TRAINING_DIR / "train.parquet", positives_only=False):
    """As-of-time training rows (step 2 output): user features from BEFORE each rating."""
    t = pq.read_table(path)
    if positives_only:
        t = t.filter(pc.equal(t.column("label"), 1))
    return {
        "userId": t.column("userId").to_numpy(),
        "user_seq": user_seq(t, catalog),
        "user_dense": user_dense(t),
        "item_idx": catalog.to_idx(t.column("movieId").to_numpy()).reshape(-1, 1),
        "item_dense": item_dense(t),
        "label": t.column("label").to_numpy().astype(np.float32),
    }


class Users:
    """Snapshot user features as of the train cutoff (what serving sees), plus each user's seen items."""

    def __init__(self, catalog, path=TRAINING_DIR / "users.parquet", train_path=TRAINING_DIR / "train.parquet"):
        t = pq.read_table(path)
        self.user_ids = t.column("userId").to_numpy()
        self.row = {u: i for i, u in enumerate(self.user_ids)}
        self.seq = user_seq(t, catalog)
        self.dense = user_dense(t)
        seen = pq.read_table(train_path, columns=["userId", "movieId"]).to_pandas()
        seen["idx"] = catalog.to_idx(seen["movieId"].to_numpy())
        self.seen = {u: g.to_numpy() for u, g in seen.groupby("userId")["idx"]}

    def __contains__(self, user_id):
        return user_id in self.row

    def features(self, user_ids):
        rows = np.array([self.row[u] for u in user_ids])
        return {"user_seq": self.seq[rows], "user_dense": self.dense[rows]}


def load_ratings(name):
    """val / test / cold_start ratings for the sampled users -> pandas (userId, movieId, rating, label, ts)."""
    return pq.read_table(TRAINING_DIR / f"{name}.parquet").to_pandas()
