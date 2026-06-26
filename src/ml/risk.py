"""Risk aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from config import CRITICAL_THRESHOLD, HIGH_THRESHOLD, MEDIUM_THRESHOLD


@dataclass(frozen=True)
class RiskResult:
    """Aggregated risk score."""

    email_probability: float | None
    url_probability: float | None
    overall_probability: float
    level: str
    prediction: str


def risk_level(probability: float) -> str:
    """Convert probability to a display risk level."""
    if probability >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def aggregate_risk(
    email_probability: float | None = None,
    url_probability: float | None = None,
) -> RiskResult:
    """Combine available email and URL probabilities into one risk score."""
    available = [
        probability
        for probability in (email_probability, url_probability)
        if probability is not None
    ]
    overall = max(available) if available else 0.0
    level = risk_level(overall)
    return RiskResult(
        email_probability=email_probability,
        url_probability=url_probability,
        overall_probability=overall,
        level=level,
        prediction="Phishing" if overall >= MEDIUM_THRESHOLD else "Legitimate",
    )
