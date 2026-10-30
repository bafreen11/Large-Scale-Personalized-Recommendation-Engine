"""Step 5: deep ranking models that re-score the retrieved candidates. Both output a logit for P(rating >= 4).

    DCN        DCN-v2: two explicit cross layers (learned feature interactions) + a deep MLP
    WideDeep   Wide & Deep: a linear "memorisation" part (item bias, user-genre x item-genre crosses)
               + a deep MLP "generalisation" part
They share the same inputs (src.training.data feature contract), so they are interchangeable in serving
and in the A/B test.
"""
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.common.config import N_GENRES, SEQ_LEN
from src.retrieval.two_tower import history_pool
from src.training.data import ITEM_DENSE, USER_DENSE


def mlp(*units):
    return tf.keras.Sequential([tf.keras.layers.Dense(u, "relu") for u in units])


class RankerBase(tf.keras.Model):
    def __init__(self, n_items):
        super().__init__()
        self.item_emb = tf.keras.layers.Embedding(n_items + 1, 32)

    def genre_cross(self, f):
        # user genre affinity x item genres, per genre (19 values)
        return f["user_dense"][:, 3:3 + N_GENRES] * f["item_dense"][:, 1:1 + N_GENRES]

    def feature_vector(self, f):
        hist = history_pool(self.item_emb, f["user_seq"])
        item = self.item_emb(f["item_idx"][:, 0])
        return tf.concat([hist, item, hist * item, f["user_dense"], f["item_dense"], self.genre_cross(f)], 1)


class DCN(RankerBase):
    def __init__(self, n_items):
        super().__init__(n_items)
        self.cross1, self.cross2 = tfrs.layers.dcn.Cross(), tfrs.layers.dcn.Cross()
        self.deep = mlp(256, 128)
        self.out = tf.keras.layers.Dense(1)

    def call(self, f):
        x0 = self.feature_vector(f)
        x = self.cross2(x0, self.cross1(x0))
        return self.out(self.deep(x))


class WideDeep(RankerBase):
    def __init__(self, n_items):
        super().__init__(n_items)
        self.item_bias = tf.keras.layers.Embedding(n_items + 1, 1)
        self.wide = tf.keras.layers.Dense(1)
        self.deep = tf.keras.Sequential([mlp(256, 128), tf.keras.layers.Dense(1)])

    def call(self, f):
        wide_x = tf.concat([self.genre_cross(f), f["user_dense"], f["item_dense"]], 1)
        wide = self.wide(wide_x) + self.item_bias(f["item_idx"][:, 0])
        return wide + self.deep(self.feature_vector(f))


RANKERS = {"dcn": DCN, "wide_deep": WideDeep}


def export_ranker(model, path):
    """SavedModel with named inputs/outputs; the same signature is served by Triton."""
    @tf.function(input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int32, name="user_seq"),
                                  tf.TensorSpec([None, USER_DENSE], tf.float32, name="user_dense"),
                                  tf.TensorSpec([None, 1], tf.int32, name="item_idx"),
                                  tf.TensorSpec([None, ITEM_DENSE], tf.float32, name="item_dense")])
    def serve(user_seq, user_dense, item_idx, item_dense):
        logit = model({"user_seq": user_seq, "user_dense": user_dense, "item_idx": item_idx, "item_dense": item_dense})
        return {"score": tf.sigmoid(logit)}

    tf.saved_model.save(model, str(path), signatures={"serving_default": serve})
