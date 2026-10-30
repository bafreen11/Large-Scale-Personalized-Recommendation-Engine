"""Offline evaluation: warm users on the val period, plus cold-start users.

Run from the project root:
    python -m src.evaluation.evaluate            (SPLIT=test for the test period, MAX_USERS=3000)

Relevant = movies the user rated >= 4 in the period. Movies seen in training are never recommended.
Warm methods: popularity | two-tower (FAISS order) | two-tower + DCN | two-tower + Wide&Deep.
candidate_recall@N = share of relevant movies the retrieval stage surfaces at all (ceiling for the rankers).
Cold-start: first 3 ratings of a new user are "onboarding"; we predict their later positives.
"""
import json
import os

import numpy as np

from src.common.config import MODELS_DIR, N_CANDIDATES, N_POPULAR_CANDIDATES, TOP_K
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.serving.recommender import VARIANTS, Recommender
from src.training.data import load_ratings

SPLIT = os.environ.get("SPLIT", "val")
MAX_USERS = int(os.environ.get("MAX_USERS", "3000"))
K = TOP_K


def relevant_sets(rec, ratings):
    pos = ratings[ratings["label"] == 1]
    out = {}
    for u, movies in pos.groupby("userId")["movieId"]:
        idx = set(rec.cat.to_idx(movies.to_numpy()).tolist()) - {0}
        if idx:
            out[u] = idx
    return out


def eval_warm(rec):
    rel = {u: s for u, s in relevant_sets(rec, load_ratings(SPLIT)).items() if u in rec.users}
    users = sorted(rel)[:MAX_USERS]
    methods = ["popularity", "two_tower"] + [f"two_tower+{v}" for v in VARIANTS]
    scores = {m: {"recall": [], "ndcg": []} for m in methods}
    cand_recall = []
    for start in range(0, len(users), 256):
        batch = users[start:start + 256]
        for u, (cand, _) in zip(batch, rec.retrieve(batch)):
            relevant = rel[u]
            lists = {"popularity": rec.cold.popular_items(K, rec.users.seen.get(u, ())), "two_tower": cand}
            mixed = rec.with_popular(u, cand)
            for v in VARIANTS:
                lists[f"two_tower+{v}"] = rec.rerank(u, mixed, v)[0]
            for m, items in lists.items():
                items = list(items[:K])
                scores[m]["recall"].append(recall_at_k(items, relevant, K))
                scores[m]["ndcg"].append(ndcg_at_k(items, relevant, K))
            cand_recall.append(len(set(mixed.tolist()) & relevant) / len(relevant))
    table = {m: {f"recall@{K}": round(float(np.mean(s["recall"])), 4), f"ndcg@{K}": round(float(np.mean(s["ndcg"])), 4)}
             for m, s in scores.items()}
    return {"split": SPLIT, "users": len(users),
            f"candidate_recall@{N_CANDIDATES}+{N_POPULAR_CANDIDATES}pop": round(float(np.mean(cand_recall)), 4),
            "methods": table}


def eval_cold(rec, n_onboard=3):
    r = load_ratings("cold_start").sort_values(["userId", "ts"])
    scores = {m: {"recall": [], "ndcg": []} for m in ["popularity", "served_fallback"]}
    for _, g in r.groupby("userId"):
        if len(g) <= n_onboard:
            continue
        first, rest = g.iloc[:n_onboard], g.iloc[n_onboard:]
        relevant = set(rec.cat.to_idx(rest.loc[rest["label"] == 1, "movieId"].to_numpy()).tolist()) - {0}
        if not relevant:
            continue
        exclude = rec.cat.to_idx(first["movieId"].to_numpy())
        liked = rec.cat.to_idx(first.loc[first["label"] == 1, "movieId"].to_numpy())
        lists = {"popularity": rec.cold.popular_items(K, exclude),
                 "served_fallback": rec.cold.recommend(K, liked, exclude)[0]}   # content if they liked something
        for m, items in lists.items():
            scores[m]["recall"].append(recall_at_k(list(items), relevant, K))
            scores[m]["ndcg"].append(ndcg_at_k(list(items), relevant, K))
    n = len(scores["popularity"]["recall"])
    return {"users": n, "methods": {m: {f"recall@{K}": round(float(np.mean(s["recall"])), 4) if n else None,
                                        f"ndcg@{K}": round(float(np.mean(s["ndcg"])), 4) if n else None}
                                    for m, s in scores.items()}}


def main():
    rec = Recommender(use_cache=False)
    report = {"warm": eval_warm(rec), "cold_start": eval_cold(rec)}
    (MODELS_DIR / f"eval_report_{SPLIT}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
