"""Tests for URL feature extraction."""

from __future__ import annotations

from src.url.features import build_url_feature_frame, extract_url_features


def test_extract_url_features_detects_ip_and_keywords() -> None:
    """URL features should detect common phishing indicators."""
    features = extract_url_features("http://192.168.1.4/verify/password")
    assert features["contains_ip"] == 1.0
    assert features["uses_http"] == 1.0
    assert features["suspicious_keyword_count"] >= 2.0


def test_build_url_feature_frame_has_expected_columns() -> None:
    """Feature frames should preserve stable column names."""
    frame = build_url_feature_frame(["https://example.com/login"])
    assert "url_length" in frame.columns
    assert len(frame) == 1
