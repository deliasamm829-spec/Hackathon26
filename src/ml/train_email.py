"""Training pipeline for the email phishing detector."""

from __future__ import annotations

import logging

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import (
    EMAIL_METRICS_FILE,
    EMAIL_MODEL_FILE,
    EMAIL_VECTORIZER_FILE,
    FIGURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_directories,
)
from src.ml.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from src.preprocessing.text import clean_email_text
from src.utils.data_loading import load_email_dataset


LOGGER = logging.getLogger(__name__)


def train_email_model() -> dict[str, object]:
    """Train Logistic Regression and Naive Bayes, save the best email model."""
    ensure_directories()
    data = load_email_dataset()
    data["clean_text"] = data["text"].map(clean_email_text)
    data = data[data["clean_text"].str.strip() != ""]
    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"] if data["label"].nunique() > 1 else None,
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "multinomial_naive_bayes": MultinomialNB(),
    }
    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_f1 = -1.0
    for name, model in candidates.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = evaluate_binary_classifier(
            y_test.to_numpy(),
            predictions,
            probabilities,
        )
        results[name] = metrics
        if metrics["f1_score"] > best_f1:
            best_name = name
            best_pipeline = pipeline
            best_f1 = float(metrics["f1_score"])
    if best_pipeline is None:
        raise RuntimeError("Email model training failed.")
    vectorizer = best_pipeline.named_steps["tfidf"]
    model = best_pipeline.named_steps["model"]
    joblib.dump(model, EMAIL_MODEL_FILE)
    joblib.dump(vectorizer, EMAIL_VECTORIZER_FILE)
    best_metrics = results[best_name]
    best_metrics["selected_model"] = best_name
    save_metrics({"models": results, "best_model": best_name}, EMAIL_METRICS_FILE)
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        FIGURES_DIR / "email_confusion_matrix.png",
        "Email Confusion Matrix",
    )
    probabilities = best_pipeline.predict_proba(x_test)[:, 1]
    plot_roc_curve(y_test.to_numpy(), probabilities, FIGURES_DIR / "email_roc_curve.png")
    LOGGER.info("Saved email model to %s", EMAIL_MODEL_FILE)
    return {"best_model": best_name, "metrics": best_metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(train_email_model())
