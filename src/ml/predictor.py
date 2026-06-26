"""High-level prediction orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from config import EMAIL_MODEL_FILE, EMAIL_VECTORIZER_FILE, URL_MODEL_FILE
from src.email.detector import EmailDetector, EmailPrediction
from src.ml.risk import RiskResult, aggregate_risk
from src.url.detector import UrlDetector, UrlPrediction


@dataclass(frozen=True)
class AnalysisResult:
    """Combined email and URL analysis result."""

    email: EmailPrediction | None
    url: UrlPrediction | None
    risk: RiskResult


def models_are_available() -> bool:
    """Return whether all trained model artifacts exist."""
    return EMAIL_MODEL_FILE.exists() and EMAIL_VECTORIZER_FILE.exists() and URL_MODEL_FILE.exists()


def analyze(email_text: str = "", url: str = "") -> AnalysisResult:
    """Analyze an email, a URL, or both."""
    email_result: EmailPrediction | None = None
    url_result: UrlPrediction | None = None
    if email_text.strip():
        email_result = EmailDetector().predict(email_text)
    if url.strip():
        url_result = UrlDetector().predict(url)
    risk = aggregate_risk(
        email_result.probability if email_result else None,
        url_result.probability if url_result else None,
    )
    return AnalysisResult(email=email_result, url=url_result, risk=risk)
