"""Dataset loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import EMAIL_DATASET_CANDIDATES, MAX_TRAIN_ROWS, URL_DATASET_CANDIDATES


LOGGER = logging.getLogger(__name__)


def existing_paths(candidates: tuple[Path, ...]) -> list[Path]:
    """Return candidate paths that exist on disk."""
    return [path for path in candidates if path.exists()]


def _label_column(frame: pd.DataFrame) -> str | None:
    for column in ("is_phishing", "label", "target", "class"):
        if column in frame.columns:
            return column
    return None


def _compose_email_text(frame: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for column in ("subject", "body", "sender", "sender_email", "receiver", "urls"):
        if column in frame.columns:
            parts.append(frame[column].fillna("").astype(str))
    if not parts:
        raise ValueError("No usable email text columns were found.")
    text = parts[0]
    for part in parts[1:]:
        text = text + " " + part
    return text


def load_email_dataset(max_rows: int = MAX_TRAIN_ROWS) -> pd.DataFrame:
    """Load and normalize all available email datasets."""
    frames: list[pd.DataFrame] = []
    for path in existing_paths(EMAIL_DATASET_CANDIDATES):
        LOGGER.info("Loading email dataset from %s", path)
        frame = pd.read_csv(path, nrows=max_rows)
        label_col = _label_column(frame)
        if label_col is None:
            LOGGER.warning("Skipping %s because it has no label column.", path)
            continue
        normalized = pd.DataFrame(
            {
                "text": _compose_email_text(frame),
                "label": frame[label_col].fillna(0).astype(int).clip(0, 1),
                "source": path.name,
            }
        )
        frames.append(normalized.dropna(subset=["text", "label"]))
    if not frames:
        raise FileNotFoundError(
            "No email datasets found. Place SpamAssasin.csv, CEAS_08.csv or a CSV "
            "with text/label columns in data/raw/."
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text", "label"])


def load_url_dataset(max_rows: int = MAX_TRAIN_ROWS) -> pd.DataFrame:
    """Load a URL dataset, or derive one from labeled email datasets when needed."""
    frames: list[pd.DataFrame] = []
    for path in existing_paths(URL_DATASET_CANDIDATES):
        LOGGER.info("Loading URL dataset from %s", path)
        frame = pd.read_csv(path, nrows=max_rows)
        url_col = next((col for col in ("url", "urls", "URL") if col in frame.columns), None)
        label_col = _label_column(frame)
        if url_col and label_col:
            frames.append(
                pd.DataFrame(
                    {
                        "url": frame[url_col].fillna("").astype(str),
                        "label": frame[label_col].fillna(0).astype(int).clip(0, 1),
                    }
                )
            )
    if frames:
        return pd.concat(frames, ignore_index=True).query("url != ''")
    email_frame = load_email_dataset(max_rows=max_rows)
    extracted = email_frame[email_frame["text"].str.contains("http|www\\.", case=False, na=False)]
    rows: list[dict[str, object]] = []
    for _, row in extracted.iterrows():
        for token in str(row["text"]).split():
            if token.startswith(("http://", "https://", "www.")):
                rows.append({"url": token.strip(" ,;.'\"<>"), "label": int(row["label"])})
    if rows:
        return pd.DataFrame(rows).drop_duplicates()
    return synthetic_url_dataset()


def synthetic_url_dataset() -> pd.DataFrame:
    """Provide a small deterministic fallback dataset for demos and tests."""
    samples = [
        ("https://www.university.edu/portal", 0),
        ("https://github.com/login", 0),
        ("https://www.paypal.com/signin", 0),
        ("https://bank.example.com/account/summary", 0),
        ("http://192.168.1.55/verify-password", 1),
        ("http://paypal-alert-security.com/verify/login", 1),
        ("http://free-gift-wallet.example.ru/confirm", 1),
        ("https://secure-account-update.example.net/password?token=9999", 1),
    ]
    return pd.DataFrame(samples, columns=["url", "label"])
