"""Tests for prediction classes with tiny trained models."""

from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.email.detector import EmailDetector
from src.url.detector import UrlDetector
from src.url.features import build_url_feature_frame


def test_email_detector_predicts_probability(tmp_path) -> None:
    """Email detector should return a valid probability."""
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(["hello meeting", "verify password urgent"])
    model = LogisticRegression().fit(x, [0, 1])
    model_path = tmp_path / "email_model.joblib"
    vectorizer_path = tmp_path / "email_vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    result = EmailDetector(model_path, vectorizer_path).predict("verify password now")
    assert 0.0 <= result.probability <= 1.0


def test_url_detector_predicts_probability(tmp_path) -> None:
    """URL detector should return a valid probability."""
    urls = ["https://example.com", "http://192.168.1.1/verify-password"]
    x = build_url_feature_frame(urls)
    model = LogisticRegression().fit(x, [0, 1])
    model_path = tmp_path / "url_model.joblib"
    joblib.dump(model, model_path)
    result = UrlDetector(model_path).predict("http://192.168.1.1/verify-password")
    assert 0.0 <= result.probability <= 1.0
