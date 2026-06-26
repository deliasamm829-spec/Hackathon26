"""Figure generation for model reporting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from config import (
    EMAIL_METRICS_FILE,
    EMAIL_MODEL_FILE,
    EMAIL_VECTORIZER_FILE,
    FIGURES_DIR,
    URL_METRICS_FILE,
    URL_MODEL_FILE,
    ensure_directories,
)
from src.preprocessing.text import clean_email_text
from src.url.features import feature_names
from src.utils.data_loading import load_email_dataset


def generate_wordcloud(output_path: Path = FIGURES_DIR / "email_wordcloud.png") -> Path:
    """Generate a word cloud from the available email corpus."""
    ensure_directories()
    data = load_email_dataset()
    text = " ".join(data["text"].astype(str).map(clean_email_text).tolist())
    if not text.strip():
        text = "phishing detection email security"
    cloud = WordCloud(width=1200, height=700, background_color="white").generate(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_file(str(output_path))
    return output_path


def generate_email_feature_importance(
    output_path: Path = FIGURES_DIR / "email_feature_importance.png",
    top_n: int = 20,
) -> Path | None:
    """Plot the highest positive email model coefficients."""
    if not EMAIL_MODEL_FILE.exists() or not EMAIL_VECTORIZER_FILE.exists():
        return None
    model = joblib.load(EMAIL_MODEL_FILE)
    vectorizer = joblib.load(EMAIL_VECTORIZER_FILE)
    if not hasattr(model, "coef_"):
        return None
    features = pd.DataFrame(
        {
            "feature": vectorizer.get_feature_names_out(),
            "weight": model.coef_[0],
        }
    ).sort_values("weight", ascending=False)
    top = features.head(top_n).sort_values("weight")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["feature"], top["weight"], color="#2e86de")
    axis.set_title("Top Email Phishing Indicators")
    axis.set_xlabel("Model Weight")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_url_feature_importance(
    output_path: Path = FIGURES_DIR / "url_feature_importance.png",
) -> Path | None:
    """Plot URL model feature coefficients."""
    if not URL_MODEL_FILE.exists():
        return None
    model = joblib.load(URL_MODEL_FILE)
    if not hasattr(model, "coef_"):
        return None
    importance = pd.DataFrame(
        {
            "feature": feature_names(),
            "weight": model.coef_[0],
        }
    ).sort_values("weight")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(importance["feature"], importance["weight"], color="#00a676")
    axis.set_title("URL Feature Importance")
    axis.set_xlabel("Model Weight")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def load_metrics(path: Path) -> dict[str, object]:
    """Read a metrics JSON file if it exists."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def generate_all_reports() -> dict[str, object]:
    """Generate every available static report figure."""
    return {
        "wordcloud": str(generate_wordcloud()),
        "email_feature_importance": str(generate_email_feature_importance()),
        "url_feature_importance": str(generate_url_feature_importance()),
        "email_metrics": load_metrics(EMAIL_METRICS_FILE),
        "url_metrics": load_metrics(URL_METRICS_FILE),
    }


if __name__ == "__main__":
    print(generate_all_reports())
