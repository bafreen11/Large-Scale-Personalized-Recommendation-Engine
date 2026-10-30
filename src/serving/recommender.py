"""The full recommendation path used by the API, the offline evaluation and the A/B test.

    known user:  user tower -> FAISS top candidates (+ trending movies) -> drop already-seen -> ranker re-scores -> top k
    new user:    cold-start fallback (popularity, or content-based if they told us movies they like)
    every list is cached (Redis or in-memory) per (variant, user, k)
"""
import zlib

import faiss
import numpy as np

from src.common.config import MODELS_DIR, N_CANDIDATES, N_POPULAR_CANDIDATES, TOP_K
from src.index.build_faiss import query_vectors
from src.serving.cache import RecCache
from src.serving.cold_start import ColdStart
from src.serving.models import load_model
from src.training.data import Catalog, Users

VARIANTS = ("dcn", "wide_deep")


def ab_variant(user_id):
    """Stable 50/50 A/B assignment: the same user always gets the same ranker."""
    return VARIANTS[zlib.crc32(str(user_id).encode()) % 2]


class Recommender:
    def __init__(self, use_cache=True):
        self.cat = Catalog()
        self.users = Users(self.cat)
        self.user_tower = load_model("user_tower", MODELS_DIR / "two_tower" / "user_tower")
        self.rankers = {v: load_model(f"ranker_{v}", MODELS_DIR / f"ranker_{v}") for v in VARIANTS}
        self.index = faiss.read_index(str(MODELS_DIR / "faiss" / "items_ivf.faiss"))
        self.cold = ColdStart(self.cat)
        self.cache = RecCache() if use_cache else None

    # ---- building blocks ----
    def retrieve(self, user_ids, n=N_CANDIDATES):
        """FAISS candidates per user, already-seen movies removed. Returns list of (item_idx, score)."""
        emb = query_vectors(self.user_tower(self.users.features(user_ids)))
        scores, idx = self.index.search(emb, n * 3)          # over-fetch, then drop seen
        out = []
        for u, row_i, row_s in zip(user_ids, idx, scores):
            keep = (row_i > 0) & ~np.isin(row_i, self.users.seen.get(u, ()))
            out.append((row_i[keep][:n].astype(np.int32), row_s[keep][:n]))
        return out

    def with_popular(self, user_id, cand, n=N_POPULAR_CANDIDATES):
        """Second candidate source: trending movies the user hasn't seen. The ranker decides between both."""
        pop = self.cold.popular_items(n, np.concatenate([cand, self.users.seen.get(user_id, np.array([], np.int32))]))
        return np.concatenate([cand, pop]).astype(np.int32)

    def rerank(self, user_id, cand, variant):
        f = self.users.features([user_id])
        feats = {"user_seq": np.repeat(f["user_seq"], len(cand), 0),
                 "user_dense": np.repeat(f["user_dense"], len(cand), 0),
                 **self.cat.features(cand)}
        s = self.rankers[variant](feats).ravel()
        order = np.argsort(-s)
        return cand[order], s[order]

    # ---- public API ----
    def recommend(self, user_id, k=TOP_K, variant="dcn", liked_movie_ids=()):
        """Returns {"items": [{movieId, title, score}], "source": ..., "cached": bool}."""
        key = RecCache.key(variant, user_id, k)
        if self.cache is not None and not liked_movie_ids:
            hit = self.cache.get(key)
            if hit is not None:
                return {**hit, "cached": True}

        if user_id in self.users:
            cand, _ = self.retrieve([user_id])[0]
            idx, scores = self.rerank(user_id, self.with_popular(user_id, cand), variant)
            idx, scores, source = idx[:k], scores[:k], f"two_tower+{variant}"
        else:
            liked = self.cat.to_idx(list(liked_movie_ids)) if liked_movie_ids else ()
            idx, source = self.cold.recommend(k, liked, exclude=liked)
            scores = np.zeros(len(idx))

        result = {"items": [{"movieId": int(self.cat.movie_ids[i - 1]), "title": str(self.cat.titles[i - 1]),
                             "score": round(float(s), 4)} for i, s in zip(idx, scores)],
                  "source": source}
        if self.cache is not None and not liked_movie_ids:
            self.cache.set(key, result)
        return {**result, "cached": False}
