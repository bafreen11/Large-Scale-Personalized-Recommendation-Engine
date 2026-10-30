"""Run ONCE before the step 2 Spark job:  python -m src.features.genome_pca
Does the genome PCA with numpy (a ~14k x 1128 matrix is tiny) instead of inside Spark,
so the heavy Spark job never has to do it. Output: data/features/genome_pca.parquet"""
import os
import numpy as np
from src.features import config as C


def pca_numpy(X, k):
    mu = X.mean(axis=0)
    Xc = X - mu                                   # centre (PCA needs this)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:k].T                             # (n_movies, k)
    explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
    return Z, explained


def main():
    import pyarrow as pa, pyarrow.dataset as ds, pyarrow.parquet as pq
    t = ds.dataset(C.GENOME_PATH).to_table(columns=["movieId", "tagId", "relevance"])
    mid = t["movieId"].to_numpy().astype(np.int64)
    tag = t["tagId"].to_numpy().astype(np.int64)
    rel = t["relevance"].to_numpy().astype(np.float64)

    movie_ids, inv = np.unique(mid, return_inverse=True)
    X = np.zeros((len(movie_ids), C.N_GENOME_TAGS), dtype=np.float64)
    X[inv, tag - 1] = rel                         # tagId is 1-based
    print(f"matrix: {X.shape[0]:,} movies x {X.shape[1]} tags")

    Z, explained = pca_numpy(X, C.PCA_K)
    print(f"PCA k={C.PCA_K} explains {explained:.1%} of variance")

    k = C.PCA_K
    arr = pa.ListArray.from_arrays(pa.array(np.arange(0, len(Z) * k + 1, k, dtype=np.int32)),
                                   pa.array(Z.ravel()))
    out = pa.table({"movieId": pa.array(movie_ids.astype(np.int32)), "genome_pca": arr})
    os.makedirs(C.OUT_DIR, exist_ok=True)
    pq.write_table(out, C.GENOME_PCA_OUT)
    print("wrote", C.GENOME_PCA_OUT)


if __name__ == "__main__":
    main()
