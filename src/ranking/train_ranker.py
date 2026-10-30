"""Step 5: train the ranking models on rated (user, movie) pairs, label = rating >= 4.

Run from the project root:
    python -m src.ranking.train_ranker                 (both models; RANKERS=dcn to train one, RANK_EPOCHS=2)

Training rows use as-of-time user features (step 2). Validation uses the val period with the
snapshot user/item features, exactly what serving sees. Outputs: models/ranker_<name>/ SavedModel.
"""
import json
import os

import numpy as np
import tensorflow as tf

from src.common.config import MODELS_DIR, SEED
from src.ranking.models import RANKERS, export_ranker
from src.training.data import Catalog, Users, load_interactions, load_ratings

EPOCHS = int(os.environ.get("RANK_EPOCHS", "2"))
BATCH = 4096
KEYS = ["user_seq", "user_dense", "item_idx", "item_dense"]


def val_features(cat, users):
    r = load_ratings("val")
    r = r[r["userId"].isin(users.row) & (cat.to_idx(r["movieId"].to_numpy()) > 0)]
    f = users.features(r["userId"].to_numpy())
    f.update(cat.features(cat.to_idx(r["movieId"].to_numpy())))
    return f, r["label"].to_numpy().astype(np.float32)


def main():
    tf.keras.utils.set_random_seed(SEED)
    cat = Catalog()
    users = Users(cat)
    train = load_interactions(cat)
    val_x, val_y = val_features(cat, users)
    print(f"train rows={len(train['label']):,} (pos {train['label'].mean():.1%})   val rows={len(val_y):,}")

    train_ds = (tf.data.Dataset.from_tensor_slices(({k: train[k] for k in KEYS}, train["label"]))
                .shuffle(200_000, seed=SEED).batch(BATCH).prefetch(2))
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH)

    report = {}
    for name in os.environ.get("RANKERS", "dcn,wide_deep").split(","):
        model = RANKERS[name](len(cat))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.AUC(from_logits=True, name="auc")])
        print(f"--- {name}")
        hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
        export_ranker(model, MODELS_DIR / f"ranker_{name}")
        report[name] = {"val_auc": round(hist.history["val_auc"][-1], 4), "val_loss": round(hist.history["val_loss"][-1], 4)}

    (MODELS_DIR / "ranker_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
