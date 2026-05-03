from pathlib import Path
import argparse

from language_detector.char_cnn import train_cnn
from language_detector.data_utils import load_jsonl
from language_detector.tfidf_baseline import train_tfidf


DATA_DIR = Path("data") / "processed"
TFIDF_MODEL_PATH = Path("models") / "tfidf" / "language_detector.joblib"
CNN_MODEL_PATH = Path("models") / "cnn" / "char_cnn.pt"


def main():
    parser = argparse.ArgumentParser(description="Train the code language detector.")
    parser.add_argument(
        "--model",
        choices=("tfidf", "cnn"),
        default="tfidf",
        help="Which model to train.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder containing train.jsonl and validation.jsonl.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--snippet-len",
        type=int,
        default=200,
        help="Randomly crop training code to this many characters. Use -1 to disable.",
    )
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per split.")
    args = parser.parse_args()

    snippet_len = None if args.snippet_len == -1 else args.snippet_len
    model_path = args.model_path
    if model_path is None:
        model_path = TFIDF_MODEL_PATH if args.model == "tfidf" else CNN_MODEL_PATH

    train_x, train_y = load_jsonl(
        args.data_dir / "train.jsonl",
        snippet_len=snippet_len,
        limit=args.limit,
    )
    valid_x, valid_y = load_jsonl(
        args.data_dir / "validation.jsonl",
        snippet_len=snippet_len,
        limit=args.limit,
    )

    if args.model == "tfidf":
        train_tfidf(train_x, train_y, valid_x, valid_y, model_path)
    else:
        train_cnn(
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            checkpoint_path=model_path,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
