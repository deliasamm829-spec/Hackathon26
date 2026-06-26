"""Central configuration for the phishing detection project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
HISTORY_FILE = REPORTS_DIR / "analysis_history.csv"

EMAIL_MODEL_FILE = MODELS_DIR / "email_model.joblib"
EMAIL_VECTORIZER_FILE = MODELS_DIR / "email_vectorizer.joblib"
EMAIL_METRICS_FILE = MODELS_DIR / "email_metrics.json"
URL_MODEL_FILE = MODELS_DIR / "url_model.joblib"
URL_METRICS_FILE = MODELS_DIR / "url_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TRAIN_ROWS = 30000

LOW_THRESHOLD = 0.30
MEDIUM_THRESHOLD = 0.55
HIGH_THRESHOLD = 0.75
CRITICAL_THRESHOLD = 0.90

SUSPICIOUS_KEYWORDS = (
    "account",
    "alert",
    "bank",
    "billing",
    "confirm",
    "free",
    "invoice",
    "login",
    "password",
    "paypal",
    "secure",
    "security",
    "update",
    "urgent",
    "verify",
    "wallet",
    "winner",
)

EMAIL_DATASET_CANDIDATES = (
    RAW_DATA_DIR / "SpamAssasin.csv",
    RAW_DATA_DIR / "CEAS_08.csv",
    RAW_DATA_DIR / "phishing_email_detection_2026_dataset.csv",
    PROJECT_ROOT / "SpamAssasin.csv",
    PROJECT_ROOT / "CEAS_08.csv",
    PROJECT_ROOT / "phishing_email_detection_2026_dataset.csv",
)

URL_DATASET_CANDIDATES = (
    RAW_DATA_DIR / "urls.csv",
    RAW_DATA_DIR / "phishtank.csv",
    RAW_DATA_DIR / "PhishTank.csv",
)


def ensure_directories() -> None:
    """Create all runtime directories required by the application."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DOCS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
