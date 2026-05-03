import time
from pathlib import Path

from language_detector.char_cnn import load_cnn_checkpoint, predict_cnn
from language_detector.tfidf_baseline import load_tfidf_model, predict_tfidf


TFIDF_MODEL_PATH = Path("models") / "tfidf" / "language_detector.joblib"
LEGACY_TFIDF_MODEL_PATH = Path("models") / "language_detector.joblib"
CNN_MODEL_PATH = Path("models") / "cnn" / "char_cnn.pt"


class InferenceService:
    def __init__(self):
        self._models = {}

    def load_models(self):
        self._models.clear()
        self._load_tfidf()
        self._load_cnn()

    def clear(self):
        self._models.clear()

    def available_models(self):
        return sorted(self._models.keys())

    def has_model(self, model_name):
        return model_name in self._models

    def predict(self, model_name, code):
        start = time.perf_counter()
        language, confidence = self._models[model_name](code)
        inference_time_ms = (time.perf_counter() - start) * 1000

        return {
            "model": model_name,
            "language": language,
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
        }

    def compare(self, code):
        return [self.predict(model_name, code) for model_name in self.available_models()]

    def _load_tfidf(self):
        model_path = self._existing_tfidf_path()
        if model_path is None:
            return

        model = load_tfidf_model(model_path)
        self._models["tfidf"] = lambda code: predict_tfidf(model, code)

    def _load_cnn(self):
        if not CNN_MODEL_PATH.exists():
            return

        model, checkpoint, device = load_cnn_checkpoint(CNN_MODEL_PATH)
        self._models["cnn"] = lambda code: predict_cnn(model, checkpoint, device, code)

    @staticmethod
    def _existing_tfidf_path():
        if TFIDF_MODEL_PATH.exists():
            return TFIDF_MODEL_PATH
        if LEGACY_TFIDF_MODEL_PATH.exists():
            return LEGACY_TFIDF_MODEL_PATH
        return None
