import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def build_tfidf_model():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=200_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_tfidf(train_x, train_y, valid_x, valid_y, model_path):
    model = build_tfidf_model()
    model.fit(train_x, train_y)

    predictions = model.predict(valid_x)
    print(classification_report(valid_y, predictions))
    print(f"Validation accuracy: {accuracy_score(valid_y, predictions):.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved TF-IDF model to {model_path}")


def load_tfidf_model(model_path):
    return joblib.load(model_path)


def predict_tfidf(model, code):
    language = model.predict([code])[0]
    probabilities = model.predict_proba([code])[0]
    confidence = dict(zip(model.classes_, probabilities))[language]
    return language, confidence
