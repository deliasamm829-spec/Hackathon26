"""Training pipeline for the URL phishing detector."""

from __future__ import annotations

import logging

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import (
    FIGURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    URL_METRICS_FILE,
    URL_MODEL_FILE,
    ensure_directories,
)
from src.ml.evaluation import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from src.url.features import build_url_feature_frame
from src.utils.data_loading import load_url_dataset, synthetic_url_dataset


LOGGER = logging.getLogger(__name__)


def train_url_model() -> dict[str, object]:
    """Train and save the URL phishing model."""
    ensure_directories()
    data = load_url_dataset()
    data = data[data["url"].astype(str).str.strip() != ""]
    if data["label"].nunique() < 2:
        fallback = load_url_dataset(max_rows=0)
        data = fallback if not fallback.empty else synthetic_url_dataset()
        if data["label"].nunique() < 2:
            data = synthetic_url_dataset()
    x = build_url_feature_frame(data["url"].astype(str).tolist())
    y = data["label"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 and y.value_counts().min() > 1 else None,
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_binary_classifier(y_test.to_numpy(), predictions, probabilities)
    save_metrics(metrics, URL_METRICS_FILE)
    joblib.dump(model, URL_MODEL_FILE)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / "url_confusion_matrix.png",
        "URL Confusion Matrix",
    )
    plot_roc_curve(y_test.to_numpy(), probabilities, FIGURES_DIR / "url_roc_curve.png")
    LOGGER.info("Saved URL model to %s", URL_MODEL_FILE)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(train_url_model())
