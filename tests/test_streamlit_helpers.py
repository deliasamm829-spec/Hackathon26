"""Tests for Streamlit helper functions."""

from __future__ import annotations

from app.helpers import probability


def test_probability_formatter() -> None:
    """Probability formatting should be user-facing and stable."""
    assert probability(0.1234) == "12.3%"
    assert probability(None) == "N/A"
