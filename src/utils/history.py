"""Local analysis history storage."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import HISTORY_FILE, ensure_directories


HISTORY_COLUMNS = [
    "date",
    "email_score",
    "url_score",
    "final_score",
    "prediction",
    "risk_level",
]


def load_history() -> pd.DataFrame:
    """Load local analysis history."""
    ensure_directories()
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(HISTORY_FILE)


def append_history(
    email_score: float | None,
    url_score: float | None,
    final_score: float,
    prediction: str,
    risk_level: str,
) -> None:
    """Append one analysis record to the history CSV."""
    ensure_directories()
    row = pd.DataFrame(
        [
            {
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "email_score": email_score,
                "url_score": url_score,
                "final_score": final_score,
                "prediction": prediction,
                "risk_level": risk_level,
            }
        ],
        columns=HISTORY_COLUMNS,
    )
    history = load_history()
    pd.concat([history, row], ignore_index=True).to_csv(HISTORY_FILE, index=False)
