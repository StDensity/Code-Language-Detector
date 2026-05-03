import argparse
import time
from pathlib import Path

from language_detector.char_cnn import load_cnn_checkpoint, predict_cnn
from language_detector.tfidf_baseline import load_tfidf_model, predict_tfidf


TFIDF_MODEL_PATH = Path("models") / "tfidf" / "language_detector.joblib"
LEGACY_TFIDF_MODEL_PATH = Path("models") / "language_detector.joblib"
CNN_MODEL_PATH = Path("models") / "cnn" / "char_cnn.pt"


def existing_tfidf_path():
    if TFIDF_MODEL_PATH.exists():
        return TFIDF_MODEL_PATH
    return LEGACY_TFIDF_MODEL_PATH


def timed_prediction(name, predict_fn):
    start_time = time.perf_counter()
    language, confidence = predict_fn()
    inference_time = time.perf_counter() - start_time
    print(f"{name}: {language} ({confidence:.2%}) in {inference_time:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Predict the language of a code snippet.")
    parser.add_argument("code", help="Code snippet to classify.")
    parser.add_argument(
        "--model",
        choices=("tfidf", "cnn", "compare"),
        default="tfidf",
        help="Which model to use.",
    )
    parser.add_argument(
        "--tfidf-model-path",
        type=Path,
        default=None,
        help="Path to the TF-IDF baseline model.",
    )
    parser.add_argument(
        "--cnn-model-path",
        type=Path,
        default=CNN_MODEL_PATH,
        help="Path to the CNN checkpoint.",
    )
    args = parser.parse_args()

    if args.model in ("tfidf", "compare"):
        tfidf_path = args.tfidf_model_path or existing_tfidf_path()
        tfidf_model = load_tfidf_model(tfidf_path)
        timed_prediction("tfidf", lambda: predict_tfidf(tfidf_model, args.code))

    if args.model in ("cnn", "compare"):
        cnn_model, cnn_checkpoint, device = load_cnn_checkpoint(args.cnn_model_path)
        timed_prediction(
            "cnn",
            lambda: predict_cnn(cnn_model, cnn_checkpoint, device, args.code),
        )


if __name__ == "__main__":
    main()
