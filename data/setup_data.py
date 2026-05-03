import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "claudios/code_search_net"
LANGUAGES = ("python", "javascript", "go", "java", "php", "ruby")
SPLITS = ("train", "validation", "test")

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "processed"


def make_split_name(split, limit):
    if limit is None:
        return split
    return f"{split}[:{limit}]"


def write_split(split, output_dir, limit):
    output_path = output_dir / f"{split}.jsonl"
    rows = []

    for language in LANGUAGES:
        dataset = load_dataset(
            DATASET_NAME,
            language,
            split=make_split_name(split, limit),
        )

        for row in dataset:
            rows.append(
                {
                    "code": row["func_code_string"],
                    "language": row["language"],
                }
            )

    random.Random(42).shuffle(rows)

    with output_path.open("w", encoding="utf-8") as output_file:
        for item in rows:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Download CodeSearchNet and create clean JSONL files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5_000,
        help="Rows per language per split. Use -1 for the full dataset.",
    )
    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        output_path, rows_written = write_split(split, OUTPUT_DIR, limit)
        print(f"{split}: wrote {rows_written} rows to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
