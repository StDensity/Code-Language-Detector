import json
import random
from pathlib import Path


def random_snippet(code, max_len):
    if max_len is None or len(code) <= max_len:
        return code

    start = random.randint(0, len(code) - max_len)
    return code[start : start + max_len]


def load_jsonl(path, snippet_len=None, limit=None):
    code_samples = []
    labels = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if limit is not None and line_number >= limit:
                break

            row = json.loads(line)
            code = random_snippet(row["code"], snippet_len)
            code_samples.append(code)
            labels.append(row["language"])

    return code_samples, labels
