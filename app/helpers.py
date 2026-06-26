"""Small helpers shared by the Streamlit application and tests."""

from __future__ import annotations


def probability(value: float | None) -> str:
    """Format probability for display."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"
