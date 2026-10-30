"""Model backends: the same call works against Triton (if TRITON_URL is set and the model is loaded there)
or a local SavedModel in this process. Inputs/outputs follow the src.training.data feature contract."""
import numpy as np
import tensorflow as tf

from src.common.config import TRITON_URL


class LocalModel:
    def __init__(self, path):
        self.fn = tf.saved_model.load(str(path)).signatures["serving_default"]
        self.backend = "local"

    def __call__(self, feats):
        out = self.fn(**{k: tf.constant(v) for k, v in feats.items()})
        return next(iter(out.values())).numpy()


class TritonModel:
    def __init__(self, name, client):
        import tritonclient.http as http
        self.http, self.client, self.name = http, client, name
        self.output = client.get_model_metadata(name)["outputs"][0]["name"]
        self.backend = "triton"

    def __call__(self, feats):
        inputs = []
        for k, v in feats.items():
            i = self.http.InferInput(k, list(v.shape), "INT32" if v.dtype == np.int32 else "FP32")
            i.set_data_from_numpy(v)
            inputs.append(i)
        return self.client.infer(self.name, inputs).as_numpy(self.output)


def load_model(name, local_path):
    """name = Triton model name (user_tower, ranker_dcn, ranker_wide_deep)."""
    if TRITON_URL:
        try:
            import tritonclient.http as http
            client = http.InferenceServerClient(TRITON_URL)
            if client.is_model_ready(name):
                return TritonModel(name, client)
            print(f"[serving] Triton has no ready model '{name}', using local SavedModel")
        except Exception as e:      # server down / not installed -> serve in-process
            print(f"[serving] Triton unavailable ({e.__class__.__name__}), using local SavedModel for {name}")
    return LocalModel(local_path)
