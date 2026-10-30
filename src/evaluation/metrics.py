"""Ranking and A/B-test metrics. `recs` is an ordered list of item ids, `relevant` a set of item ids."""
import math


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def two_proportion_ztest(clicks_a, n_a, clicks_b, n_b):
    """Is CTR_a different from CTR_b? Returns (z, two-sided p-value)."""
    p = (clicks_a + clicks_b) / (n_a + n_b)
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (clicks_a / n_a - clicks_b / n_b) / se
    return z, math.erfc(abs(z) / math.sqrt(2))
