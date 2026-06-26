"""Tests for risk aggregation."""

from __future__ import annotations

from src.ml.risk import aggregate_risk, risk_level


def test_risk_level_thresholds() -> None:
    """Risk labels should follow configured thresholds."""
    assert risk_level(0.1) == "LOW"
    assert risk_level(0.6) == "MEDIUM"
    assert risk_level(0.8) == "HIGH"
    assert risk_level(0.95) == "CRITICAL"


def test_aggregate_risk_uses_highest_available_probability() -> None:
    """Overall score should reflect the strongest available signal."""
    result = aggregate_risk(email_probability=0.2, url_probability=0.8)
    assert result.overall_probability == 0.8
    assert result.prediction == "Phishing"
