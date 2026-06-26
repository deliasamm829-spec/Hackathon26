"""Tests for text preprocessing."""

from __future__ import annotations

from src.preprocessing.text import clean_email_text, find_suspicious_words


def test_clean_email_text_removes_html_urls_numbers_and_punctuation() -> None:
    """Email cleaning should normalize noisy text."""
    cleaned = clean_email_text("<b>URGENT</b> verify at https://bad.test now!!! 123")
    assert "urgent" in cleaned
    assert "verify" in cleaned
    assert "https" not in cleaned
    assert "123" not in cleaned


def test_find_suspicious_words_returns_hits() -> None:
    """Suspicious keyword detection should find known risk terms."""
    assert find_suspicious_words("Please verify your password", ("verify", "login")) == [
        "verify"
    ]
