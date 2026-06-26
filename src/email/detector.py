"""Email phishing detector."""

from __future__ import annotations

from dataclasses import dataclass

import joblib

from config import EMAIL_MODEL_FILE, EMAIL_VECTORIZER_FILE, SUSPICIOUS_KEYWORDS
from src.explainability.email_explainer import WordContribution, explain_email_prediction
from src.preprocessing.text import clean_email_text, find_suspicious_words


@dataclass(frozen=True)
class EmailPrediction:
    """Email prediction result."""

    probability: float
    prediction: int
    suspicious_words: list[str]
    contributions: list[WordContribution]


class EmailDetector:
    """Load and run the trained email phishing model."""

    def __init__(
        self,
        model_path: object = EMAIL_MODEL_FILE,
        vectorizer_path: object = EMAIL_VECTORIZER_FILE,
    ) -> None:
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, email_text: str) -> EmailPrediction:
        """Predict phishing probability for an email body."""
        cleaned = clean_email_text(email_text)
        matrix = self.vectorizer.transform([cleaned])
        probability = float(self.model.predict_proba(matrix)[0, 1])
        return EmailPrediction(
            probability=probability,
            prediction=int(probability >= 0.5),
            suspicious_words=find_suspicious_words(email_text, SUSPICIOUS_KEYWORDS),
            contributions=explain_email_prediction(cleaned, self.vectorizer, self.model),
        )
