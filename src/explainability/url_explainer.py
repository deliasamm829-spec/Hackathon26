"""Explainability helpers for URL predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.url.features import extract_url_features


@dataclass(frozen=True)
class FeatureContribution:
    """A feature-level contribution to a URL prediction."""

    feature: str
    value: float
    weight: float
    contribution: float
    reason: str


REASON_MAP = {
    "url_length": "URL unusually long",
    "hostname_length": "Long hostname",
    "path_length": "Long URL path",
    "digit_count": "Too many digits",
    "dot_count": "Many dots in domain or path",
    "slash_count": "Many path separators",
    "subdomain_count": "Many subdomains",
    "query_param_count": "Many query parameters",
    "contains_at": 'Contains "@"',
    "contains_dash": 'Contains "-"',
    "contains_percent": 'Contains encoded characters',
    "contains_equal": "Contains query assignments",
    "contains_ip": "IP address detected",
    "uses_http": "Uses unencrypted HTTP",
    "uses_https": "Uses HTTPS",
    "suspicious_keyword_count": "Contains suspicious keywords",
    "entropy": "High entropy",
}


def explain_url_prediction(url: str, model: object, top_n: int = 8) -> list[FeatureContribution]:
    """Return the URL features that most increased phishing risk."""
    if not hasattr(model, "coef_"):
        return []
    features = extract_url_features(url)
    coefficients = np.asarray(model.coef_[0])
    names = list(features)
    contributions = [
        FeatureContribution(
            feature=name,
            value=float(features[name]),
            weight=float(coefficients[index]),
            contribution=float(features[name] * coefficients[index]),
            reason=REASON_MAP.get(name, name.replace("_", " ").title()),
        )
        for index, name in enumerate(names)
    ]
    return sorted(
        [item for item in contributions if item.contribution > 0],
        key=lambda item: item.contribution,
        reverse=True,
    )[:top_n]
