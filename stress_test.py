import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib


MODEL_PATH = Path("models") / "language_detector.joblib"
TEST_PATH = Path("data") / "manual_stress_tests.jsonl"


def load_tests(path):
    tests = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            tests.append(json.loads(line))

    return tests


def main():
    parser = argparse.ArgumentParser(description="Run manual stress tests.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_PATH,
        help="Path to manual JSONL stress tests.",
    )
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    tests = load_tests(args.test_path)

    category_totals = Counter()
    category_correct = Counter()
    failures = defaultdict(list)

    print("Manual stress test results")
    print("=" * 80)

    for test in tests:
        prediction = model.predict([test["code"]])[0]
        expected = test["expected"]
        category = test["category"]
        passed = prediction == expected

        category_totals[category] += 1
        if passed:
            category_correct[category] += 1
        else:
            failures[category].append(test)

        confidence_text = ""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([test["code"]])[0]
            confidence = dict(zip(model.classes_, probabilities))[prediction]
            confidence_text = f" confidence={confidence:.2%}"

        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | {category:9} | {test['name']:28} | "
            f"expected={expected:10} predicted={prediction:10}{confidence_text}"
        )

    print("=" * 80)
    total = sum(category_totals.values())
    correct = sum(category_correct.values())
    print(f"Overall: {correct}/{total} correct")

    for category in sorted(category_totals):
        print(
            f"{category}: {category_correct[category]}/"
            f"{category_totals[category]} correct"
        )

    if failures:
        print("\nFailures to inspect:")
        for category, tests_in_category in failures.items():
            for test in tests_in_category:
                print(f"- {category}: {test['name']}")


if __name__ == "__main__":
    main()
