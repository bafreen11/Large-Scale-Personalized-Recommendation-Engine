"""Step 3: two-tower retrieval model (TensorFlow Recommenders).

Run from the project root:
    python -m src.retrieval.two_tower            (TT_EPOCHS=5 by default)

Trains on positive interactions (rating >= 4) with an in-batch sampled softmax.
logQ correction (candidate_sampling_probability) stops popular movies from being over-penalised as negatives.
User features are as-of each rating (step 2), so the model never sees the future.

Outputs in models/two_tower/:
    user_tower/            SavedModel  (user_seq, user_dense) -> embedding    [also served by Triton]
    item_embeddings.npy    one row per catalogue movie, same order as item_ids.npy (movieIds)
"""
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.common.config import EMBED_DIM, MODELS_DIR, SEED, SEQ_LEN
from src.training.data import USER_DENSE, Catalog, load_interactions

OUT = MODELS_DIR / "two_tower"
EPOCHS = int(os.environ.get("TT_EPOCHS", "5"))
LOGQ = os.environ.get("LOGQ", "1") == "1"   # 0 drops the correction (tested: much worse retrieval)
TEMPERATURE = 0.05
BATCH = 2048


def history_pool(item_emb, seq):
    """Mean of the embeddings of the movies in the user's history (padding = 0 is masked out)."""
    mask = tf.cast(seq > 0, tf.float32)[..., None]
    return tf.reduce_sum(item_emb(seq) * mask, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1.0)


class UserTower(tf.keras.Model):
    def __init__(self, item_emb):
        super().__init__()
        self.item_emb = item_emb   # shared with the item tower: history and candidates live in one space
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(256, "relu"), tf.keras.layers.Dense(EMBED_DIM)])

    def call(self, f):
        x = tf.concat([history_pool(self.item_emb, f["user_seq"]), f["user_dense"]], 1)
        return tf.math.l2_normalize(self.mlp(x), -1)


class ItemTower(tf.keras.Model):
    def __init__(self, item_emb):
        super().__init__()
        self.item_emb = item_emb
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(256, "relu"), tf.keras.layers.Dense(EMBED_DIM)])

    def call(self, f):
        x = tf.concat([self.item_emb(f["item_idx"][:, 0]), f["item_dense"]], 1)
        return tf.math.l2_normalize(self.mlp(x), -1)


class TwoTower(tfrs.Model):
    def __init__(self, n_items):
        super().__init__()
        emb = tf.keras.layers.Embedding(n_items + 1, 64)
        self.user_tower, self.item_tower = UserTower(emb), ItemTower(emb)
        # cosine similarity / temperature: sharper softmax for L2-normalised embeddings
        self.task = tfrs.tasks.Retrieval(temperature=TEMPERATURE, remove_accidental_hits=True)

    def compute_loss(self, f, training=False):
        return self.task(self.user_tower(f), self.item_tower(f),
                         candidate_sampling_probability=f["item_prob"] if LOGQ else None,
                         candidate_ids=f["item_idx"][:, 0], compute_metrics=False)


def export_user_tower(tower, path):
    @tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int32, name="user_seq"),
                                  tf.TensorSpec([None, USER_DENSE], tf.float32, name="user_dense")])
    def serve(user_seq, user_dense):
        return {"embedding": tower({"user_seq": user_seq, "user_dense": user_dense})}

    tf.saved_model.save(tower, str(path), signatures={"serving_default": serve})


def main():
    tf.keras.utils.set_random_seed(SEED)
    cat = Catalog()
    data = load_interactions(cat, positives_only=True)
    idx = data["item_idx"][:, 0]
    freq = np.bincount(idx, minlength=len(cat) + 1)
    data["item_prob"] = (freq[idx] / len(idx)).astype(np.float32)
    feats = {k: data[k] for k in ["user_seq", "user_dense", "item_idx", "item_dense", "item_prob"]}
    print(f"positives={len(idx):,}  users={len(np.unique(data['userId'])):,}  items={int((freq > 0).sum()):,}")

    # hold out 2% of rows to watch the loss (ranking quality is measured in src.evaluation.evaluate)
    perm = np.random.default_rng(SEED).permutation(len(idx))
    n_val = len(idx) // 50
    part = lambda rows: {k: v[rows] for k, v in feats.items()}
    train_ds = (tf.data.Dataset.from_tensor_slices(part(perm[n_val:]))
                .shuffle(200_000, seed=SEED).batch(BATCH, drop_remainder=True).prefetch(2))
    val_ds = tf.data.Dataset.from_tensor_slices(part(perm[:n_val])).batch(BATCH, drop_remainder=True)

    model = TwoTower(len(cat))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    OUT.mkdir(parents=True, exist_ok=True)
    item_emb = model.item_tower.predict(cat.features(cat.item_idx), batch_size=8192, verbose=0)
    np.save(OUT / "item_embeddings.npy", item_emb.astype(np.float32))
    np.save(OUT / "item_ids.npy", cat.movie_ids)
    export_user_tower(model.user_tower, OUT / "user_tower")
    (OUT / "train_history.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in hist.history.items()}))
    print("saved", OUT)


if __name__ == "__main__":
    main()
