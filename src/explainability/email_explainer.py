"""Explainability helpers for email predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class WordContribution:
    """A word-level contribution to an email prediction."""

    word: str
    weight: float
    tfidf: float
    contribution: float


def explain_email_prediction(
    text: str,
    vectorizer: TfidfVectorizer,
    model: object,
    top_n: int = 10,
) -> list[WordContribution]:
    """Return the strongest word contributions for a linear email model."""
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_[0])
    elif hasattr(model, "feature_log_prob_"):
        coefficients = np.asarray(model.feature_log_prob_[1] - model.feature_log_prob_[0])
    else:
        return []
    matrix = vectorizer.transform([text])
    contributions = matrix.multiply(coefficients).toarray()[0]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    non_zero_indices = np.flatnonzero(matrix.toarray()[0])
    ranked = sorted(
        non_zero_indices,
        key=lambda index: abs(contributions[index]),
        reverse=True,
    )[:top_n]
    return [
        WordContribution(
            word=str(feature_names[index]),
            weight=float(coefficients[index]),
            tfidf=float(matrix[0, index]),
            contribution=float(contributions[index]),
        )
        for index in ranked
    ]
