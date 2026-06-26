"""Run all model training pipelines."""

from __future__ import annotations

import logging

from src.ml.train_email import train_email_model
from src.ml.train_url import train_url_model


def train_all() -> dict[str, object]:
    """Train email and URL models."""
    return {
        "email": train_email_model(),
        "url": train_url_model(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(train_all())
