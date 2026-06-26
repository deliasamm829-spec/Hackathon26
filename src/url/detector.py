"""URL phishing detector."""

from __future__ import annotations

from dataclasses import dataclass

import joblib

from config import URL_MODEL_FILE
from src.explainability.url_explainer import FeatureContribution, explain_url_prediction
from src.url.features import build_url_feature_frame


@dataclass(frozen=True)
class UrlPrediction:
    """URL prediction result."""

    probability: float
    prediction: int
    contributions: list[FeatureContribution]


class UrlDetector:
    """Load and run the trained URL phishing model."""

    def __init__(self, model_path: object = URL_MODEL_FILE) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, url: str) -> UrlPrediction:
        """Predict phishing probability for a URL."""
        features = build_url_feature_frame([url])
        probability = float(self.model.predict_proba(features)[0, 1])
        return UrlPrediction(
            probability=probability,
            prediction=int(probability >= 0.5),
            contributions=explain_url_prediction(url, self.model),
        )
