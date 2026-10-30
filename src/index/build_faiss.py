"""Step 4: index the item embeddings in FAISS for fast nearest-neighbour search.

Run from the project root (after src.retrieval.two_tower):
    python -m src.index.build_faiss

Builds an exact index (IndexFlatIP) and an approximate IVF index, then reports how many of the exact
top-N neighbours IVF finds and how long a single query takes. The IVF index is what serving uses.
Only movies seen in training are indexed: an untrained ID embedding is noise, and brand-new movies
are handled by the content-based fallback (src.serving.cold_start).
Vector ids are catalogue item indices (src.training.data.Catalog).

The two-tower is trained with logQ correction, so cos(u, v) alone under-rates popular movies. The
index stores [v / T, log q(v)] and queries are [u, 1]: the inner product is cos/T + log q, the full
softmax logit, so FAISS still does one exact max-inner-product search.
"""
import json
import os
import time

import faiss
import numpy as np

from src.common.config import MODELS_DIR, N_CANDIDATES
from src.retrieval.two_tower import TEMPERATURE
from src.serving.models import load_model
from src.training.data import Catalog, Users

OUT = MODELS_DIR / "faiss"
NPROBE = 32
POP_PRIOR = float(os.environ.get("POP_PRIOR", "0"))   # weight of log q; 0 = pure cosine (1.0 tested: no gain on val)


def item_vectors(emb, pop):
    log_q = np.log(np.maximum(pop, 1) / pop.sum())
    return np.ascontiguousarray(np.hstack([emb / TEMPERATURE, POP_PRIOR * log_q[:, None]]), dtype=np.float32)


def query_vectors(user_emb):
    return np.ascontiguousarray(np.hstack([user_emb, np.ones((len(user_emb), 1))]), dtype=np.float32)


def build_ivf(xb, ids):
    d, n = xb.shape[1], len(xb)
    nlist = int(4 * np.sqrt(n))                      # rule of thumb: ~4*sqrt(n) clusters
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(xb)
    ivf.add_with_ids(xb, ids)
    ivf.nprobe = NPROBE                              # clusters scanned per query: recall vs speed knob
    return ivf


def latency_ms(index, q):
    t = time.perf_counter()
    for row in q:
        index.search(row[None], N_CANDIDATES)
    return 1000 * (time.perf_counter() - t) / len(q)


def main():
    faiss.omp_set_num_threads(3)
    cat = Catalog()
    emb = np.load(MODELS_DIR / "two_tower" / "item_embeddings.npy")
    assert (np.load(MODELS_DIR / "two_tower" / "item_ids.npy") == cat.movie_ids).all(), "catalogue changed: retrain two_tower"
    seen = cat.pop_all > 0
    xb, ids = item_vectors(emb[seen], cat.pop_all[seen]), cat.item_idx[seen].astype(np.int64)

    flat = faiss.IndexIDMap(faiss.IndexFlatIP(xb.shape[1]))
    flat.add_with_ids(xb, ids)
    ivf = build_ivf(xb, ids)

    # real queries: user-tower embeddings of up to 2,000 users
    users = Users(cat)
    q_users = users.user_ids[:2000]
    q = query_vectors(load_model("user_tower", MODELS_DIR / "two_tower" / "user_tower")(users.features(q_users)))
    _, exact = flat.search(q, N_CANDIDATES)
    _, approx = ivf.search(q, N_CANDIDATES)
    recall = np.mean([len(np.intersect1d(a, e)) / N_CANDIDATES for a, e in zip(approx, exact)])

    stats = {"items_indexed": int(len(ids)), "dim": int(xb.shape[1]), "nlist": int(ivf.nlist), "nprobe": NPROBE,
             "pop_prior": POP_PRIOR,
             f"ivf_recall_vs_exact@{N_CANDIDATES}": round(float(recall), 4),
             "flat_ms_per_query": round(latency_ms(flat, q[:300]), 3),
             "ivf_ms_per_query": round(latency_ms(ivf, q[:300]), 3)}
    OUT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(ivf, str(OUT / "items_ivf.faiss"))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
