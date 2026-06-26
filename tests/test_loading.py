"""Tests for dataset loading helpers."""

from __future__ import annotations

import pandas as pd

from src.utils import data_loading


def test_existing_paths_filters_missing_paths(tmp_path) -> None:
    """Only existing candidate paths should be returned."""
    existing = tmp_path / "sample.csv"
    existing.write_text("text,label\nhello,0\n", encoding="utf-8")
    missing = tmp_path / "missing.csv"
    assert data_loading.existing_paths((existing, missing)) == [existing]


def test_demo_url_dataset_is_labeled() -> None:
    """Fallback URL data should contain both classes."""
    frame = data_loading.synthetic_url_dataset()
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["label"]) == {0, 1}
