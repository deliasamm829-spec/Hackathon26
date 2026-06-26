"""Streamlit interface for the phishing detection system."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    EMAIL_METRICS_FILE,
    FIGURES_DIR,
    URL_METRICS_FILE,
    ensure_directories,
)
from src.ml.predictor import analyze, models_are_available  # noqa: E402
from src.ml.reports import generate_all_reports, load_metrics  # noqa: E402
from src.ml.train_email import train_email_model  # noqa: E402
from src.ml.train_url import train_url_model  # noqa: E402
from src.utils.history import append_history, load_history  # noqa: E402
from app.helpers import probability  # noqa: E402


st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="shield",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        border: 1px solid rgba(127,127,127,.25);
        border-radius: 8px;
        padding: 1rem;
        min-height: 92px;
    }
    .risk-low {color: #14a44d; font-weight: 700;}
    .risk-medium {color: #c98000; font-weight: 700;}
    .risk-high {color: #d9480f; font-weight: 700;}
    .risk-critical {color: #d6336c; font-weight: 800;}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_model_status() -> None:
    """Render model availability status and training control."""
    if models_are_available():
        st.success("Models are ready.")
        return
    st.warning("Models are not trained yet.")
    if st.button("Train models", type="primary"):
        try:
            with st.status("Training models...", expanded=True) as status:
                st.write("Training email model...")
                train_email_model()
                st.write("Training URL model...")
                train_url_model()
                st.write("Generating report figures...")
                generate_all_reports()
                status.update(label="Training complete.", state="complete")
            st.success("Training complete. Refreshing model status.")
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.info(
                "Check that dependencies are installed and that the CSV datasets are readable."
            )


def render_home() -> None:
    """Home page."""
    st.title("AI-Powered Phishing Detection System")
    st.caption("Academic-grade email and URL phishing analysis with transparent explanations.")
    render_model_status()
    col1, col2, col3 = st.columns(3)
    history = load_history()
    col1.metric("Analyses", len(history))
    col2.metric("Average Risk", probability(history["final_score"].mean() if len(history) else None))
    col3.metric("Phishing Decisions", int((history["prediction"] == "Phishing").sum()) if len(history) else 0)


def render_analyse() -> None:
    """Analysis page."""
    st.title("Analyse")
    render_model_status()
    email_text = st.text_area("Email text", height=220)
    url = st.text_input("URL")
    if st.button("Analyse", type="primary", disabled=not models_are_available()):
        if not email_text.strip() and not url.strip():
            st.error("Enter an email, a URL, or both.")
            return
        with st.spinner("Analysing indicators..."):
            result = analyze(email_text, url)
        append_history(
            result.risk.email_probability,
            result.risk.url_probability,
            result.risk.overall_probability,
            result.risk.prediction,
            result.risk.level,
        )
        risk_class = f"risk-{result.risk.level.lower()}"
        st.subheader(result.risk.prediction)
        st.markdown(
            f'<p class="{risk_class}">{result.risk.level} risk - '
            f"{probability(result.risk.overall_probability)}</p>",
            unsafe_allow_html=True,
        )
        score_col1, score_col2, score_col3 = st.columns(3)
        score_col1.metric("Email probability", probability(result.risk.email_probability))
        score_col2.metric("URL probability", probability(result.risk.url_probability))
        score_col3.metric("Overall probability", probability(result.risk.overall_probability))
        if result.email:
            st.subheader("Email reasons")
            if result.email.suspicious_words:
                st.write("Suspicious words:", ", ".join(result.email.suspicious_words))
            st.dataframe(pd.DataFrame([item.__dict__ for item in result.email.contributions]))
        if result.url:
            st.subheader("URL reasons")
            st.dataframe(pd.DataFrame([item.__dict__ for item in result.url.contributions]))
        report = pd.DataFrame(
            [
                {
                    "prediction": result.risk.prediction,
                    "risk_level": result.risk.level,
                    "email_probability": result.risk.email_probability,
                    "url_probability": result.risk.url_probability,
                    "overall_probability": result.risk.overall_probability,
                }
            ]
        )
        st.download_button(
            "Download report",
            report.to_csv(index=False).encode("utf-8"),
            "phishing_analysis_report.csv",
            "text/csv",
        )


def render_history() -> None:
    """History page."""
    st.title("History")
    history = load_history()
    st.dataframe(history, width="stretch")
    if len(history):
        st.download_button(
            "Download history",
            history.to_csv(index=False).encode("utf-8"),
            "analysis_history.csv",
            "text/csv",
        )


def render_statistics() -> None:
    """Statistics page."""
    st.title("Statistics")
    if st.button("Regenerate reports"):
        with st.spinner("Generating figures..."):
            generate_all_reports()
    email_metrics = load_metrics(EMAIL_METRICS_FILE)
    url_metrics = load_metrics(URL_METRICS_FILE)
    if email_metrics:
        st.subheader("Email model")
        st.json(email_metrics)
    if url_metrics:
        st.subheader("URL model")
        st.json(url_metrics)
    for image_name in (
        "email_confusion_matrix.png",
        "email_roc_curve.png",
        "url_confusion_matrix.png",
        "url_roc_curve.png",
        "email_feature_importance.png",
        "url_feature_importance.png",
        "email_wordcloud.png",
    ):
        image_path = FIGURES_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name)


def render_about() -> None:
    """About page."""
    st.title("About")
    st.write(
        "This project combines NLP, deterministic URL feature engineering, supervised "
        "machine learning and local explainability to detect phishing attempts."
    )
    st.write("Models: Logistic Regression, Multinomial Naive Bayes, TF-IDF and URL indicators.")


def main() -> None:
    """Run the Streamlit app."""
    ensure_directories()
    page = st.sidebar.radio("Navigation", ["Home", "Analyse", "History", "Statistics", "About"])
    pages = {
        "Home": render_home,
        "Analyse": render_analyse,
        "History": render_history,
        "Statistics": render_statistics,
        "About": render_about,
    }
    pages[page]()


if __name__ == "__main__":
    main()
