import argparse
import time
from pathlib import Path

from language_detector.char_cnn import load_cnn_checkpoint, predict_cnn
from language_detector.data_utils import load_jsonl
from language_detector.tfidf_baseline import load_tfidf_model, predict_tfidf


DATA_DIR = Path("data") / "processed"
TFIDF_MODEL_PATH = Path("models") / "tfidf" / "language_detector.joblib"
LEGACY_TFIDF_MODEL_PATH = Path("models") / "language_detector.joblib"
CNN_MODEL_PATH = Path("models") / "cnn" / "char_cnn.pt"


def existing_tfidf_path():
    if TFIDF_MODEL_PATH.exists():
        return TFIDF_MODEL_PATH
    return LEGACY_TFIDF_MODEL_PATH


def evaluate(name, code_samples, labels, predict_fn):
    correct = 0
    total_time = 0.0

    for code, expected in zip(code_samples, labels):
        start = time.perf_counter()
        prediction, _confidence = predict_fn(code)
        total_time += time.perf_counter() - start

        if prediction == expected:
            correct += 1

    total = len(labels)
    accuracy = correct / total if total else 0.0
    avg_time = total_time / total if total else 0.0
    print(f"{name}: accuracy={accuracy:.4f} avg_inference={avg_time:.6f}s samples={total}")


def main():
    parser = argparse.ArgumentParser(description="Compare TF-IDF baseline and CNN.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--tfidf-model-path", type=Path, default=None)
    parser.add_argument("--cnn-model-path", type=Path, default=CNN_MODEL_PATH)
    args = parser.parse_args()

    valid_x, valid_y = load_jsonl(
        args.data_dir / "validation.jsonl",
        snippet_len=None,
        limit=args.limit,
    )

    tfidf_model = load_tfidf_model(args.tfidf_model_path or existing_tfidf_path())
    evaluate("tfidf", valid_x, valid_y, lambda code: predict_tfidf(tfidf_model, code))

    cnn_model, cnn_checkpoint, device = load_cnn_checkpoint(args.cnn_model_path)
    evaluate(
        "cnn",
        valid_x,
        valid_y,
        lambda code: predict_cnn(cnn_model, cnn_checkpoint, device, code),
    )


if __name__ == "__main__":
    main()
