"""Step 6: fallbacks for users the models have never seen.

    no history at all   -> most popular movies in the 30 days before the cutoff (all-time popularity breaks ties)
    a few liked movies  -> content-based: cosine similarity of genome-tag PCA + genres to the liked movies,
                           with a small popularity prior so obscure near-duplicates don't dominate
"""
import numpy as np


class ColdStart:
    def __init__(self, catalog, genre_weight=0.5, pop_weight=0.1):
        self.cat = catalog
        pop = catalog.pop_30d * 1e7 + catalog.pop_all
        self.popular = catalog.item_idx[np.argsort(-pop, kind="stable")]

        genome = catalog.genome / np.maximum(np.linalg.norm(catalog.genome, axis=1, keepdims=True), 1e-9)
        content = np.hstack([genome * catalog.has_genome[:, None], genre_weight * catalog.genres])
        self.content = content / np.maximum(np.linalg.norm(content, axis=1, keepdims=True), 1e-9)
        self.prior = pop_weight * np.log1p(catalog.pop_all) / np.log1p(catalog.pop_all.max())

    def popular_items(self, k, exclude=()):
        return self.popular[~np.isin(self.popular, exclude)][:k]

    def similar_items(self, liked_idx, k, exclude=()):
        liked_idx = np.asarray(liked_idx)
        scores = self.content @ self.content[liked_idx - 1].mean(0) + self.prior
        scores[np.concatenate([liked_idx, np.asarray(exclude, int)]).astype(int) - 1] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        return self.cat.item_idx[top[np.argsort(-scores[top])]]

    def recommend(self, k, liked_idx=(), exclude=()):
        """Returns (item indices, source name)."""
        liked_idx = [i for i in liked_idx if i > 0]
        if liked_idx:
            return self.similar_items(liked_idx, k, exclude), "cold_start_content"
        return self.popular_items(k, exclude), "cold_start_popular"
