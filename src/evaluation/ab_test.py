"""Step 7: simple online A/B test of the two rankers (DCN vs Wide & Deep), simulated by replaying the test period.

Run from the project root:
    python -m src.evaluation.ab_test             (MAX_USERS=4000)

Users are split 50/50 by a stable hash (same rule as the API). Each user gets TOP_K recommendations from
their variant through the full serving path (two-tower -> FAISS -> ranker -> cache). A recommendation is a
"click" if the user rated that movie >= 4 during the test period (replay simulation; no live traffic here).
Reports CTR = clicks / impressions, NDCG@K and Recall@K per variant, and a two-proportion z-test on CTR.
A second "returning visit" pass is served from the cache to show the Redis speed-up.
"""
import json
import os
import time

import numpy as np

from src.common.config import MODELS_DIR, TOP_K
from src.evaluation.metrics import ndcg_at_k, recall_at_k, two_proportion_ztest
from src.serving.recommender import VARIANTS, Recommender, ab_variant
from src.training.data import load_ratings

MAX_USERS = int(os.environ.get("MAX_USERS", "4000"))
K = TOP_K


def main():
    rec = Recommender(use_cache=True)
    rec.cache.clear()
    test = load_ratings("test")
    users = sorted(u for u in test["userId"].unique() if u in rec.users)[:MAX_USERS]
    pos = test[test["label"] == 1]
    rel = {u: set(rec.cat.to_idx(m.to_numpy()).tolist()) - {0} for u, m in pos.groupby("userId")["movieId"]}

    stats = {v: {"users": 0, "impressions": 0, "clicks": 0, "ndcg": [], "recall": []} for v in VARIANTS}
    t0 = time.perf_counter()
    for u in users:
        v = ab_variant(u)
        items = rec.cat.to_idx([i["movieId"] for i in rec.recommend(u, K, v)["items"]]).tolist()
        relevant = rel.get(u, set())
        s = stats[v]
        s["users"] += 1
        s["impressions"] += len(items)
        s["clicks"] += len(set(items) & relevant)
        if relevant:
            s["ndcg"].append(ndcg_at_k(items, relevant, K))
            s["recall"].append(recall_at_k(items, relevant, K))
    first_ms = 1000 * (time.perf_counter() - t0) / len(users)

    t0 = time.perf_counter()
    for u in users:                                   # returning visit -> served from cache
        rec.recommend(u, K, ab_variant(u))
    cached_ms = 1000 * (time.perf_counter() - t0) / len(users)

    a, b = (stats[v] for v in VARIANTS)
    z, p = two_proportion_ztest(a["clicks"], a["impressions"], b["clicks"], b["impressions"])
    report = {"variants": {v: {"users": s["users"], "impressions": s["impressions"], "clicks": s["clicks"],
                               "ctr": round(s["clicks"] / max(s["impressions"], 1), 5),
                               f"ndcg@{K}": round(float(np.mean(s["ndcg"])), 4) if s["ndcg"] else None,
                               f"recall@{K}": round(float(np.mean(s["recall"])), 4) if s["recall"] else None}
                           for v, s in stats.items()},
              "ctr_ztest": {"z": round(z, 3), "p_value": round(p, 4),
                            "significant_at_0.05": p < 0.05,
                            "winner": (VARIANTS[0] if z > 0 else VARIANTS[1]) if p < 0.05 else "no significant difference"},
              "latency_ms_per_request": {"uncached": round(first_ms, 2), "cached": round(cached_ms, 3),
                                         "cache_backend": rec.cache.backend}}
    (MODELS_DIR / "ab_test_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
