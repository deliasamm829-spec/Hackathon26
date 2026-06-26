"""Text preprocessing utilities for email analysis."""

from __future__ import annotations

import html
import logging
import re
import string
from functools import lru_cache

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover - handled by runtime fallback
    stopwords = None
    WordNetLemmatizer = None


LOGGER = logging.getLogger(__name__)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
TOKEN_PATTERN = re.compile(r"[a-z]+")

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Return English stopwords, falling back when NLTK data is unavailable."""
    if stopwords is None:
        return FALLBACK_STOPWORDS
    try:
        return set(stopwords.words("english"))
    except LookupError:
        LOGGER.warning("NLTK stopwords corpus is not installed; using fallback list.")
        return FALLBACK_STOPWORDS


@lru_cache(maxsize=1)
def wordnet_is_available() -> bool:
    """Return whether the WordNet corpus can be used."""
    if WordNetLemmatizer is None:
        return False
    try:
        WordNetLemmatizer().lemmatize("tests")
        return True
    except LookupError:
        LOGGER.warning("NLTK WordNet corpus is not installed; using fallback lemmatizer.")
        return False


@lru_cache(maxsize=1)
def get_lemmatizer() -> object | None:
    """Return a WordNet lemmatizer when the dependency and corpus are available."""
    if not wordnet_is_available():
        return None
    return WordNetLemmatizer()


def remove_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    return HTML_TAG_PATTERN.sub(" ", html.unescape(text))


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return URL_PATTERN.sub(" ", text)


def simple_lemma(token: str) -> str:
    """Small deterministic fallback lemmatizer for environments without WordNet."""
    for suffix in ("ing", "edly", "ed", "ies", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return f"{token[:-3]}y"
            return token[: -len(suffix)]
    return token


def lemmatize_token(token: str) -> str:
    """Lemmatize a token with WordNet, or use a deterministic fallback."""
    lemmatizer = get_lemmatizer()
    if lemmatizer is None:
        return simple_lemma(token)
    return str(lemmatizer.lemmatize(token))


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphabetic tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def clean_email_text(text: object) -> str:
    """Clean raw email text for TF-IDF vectorisation."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = remove_html(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = cleaned.lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\d+", " ", cleaned)
    tokens = tokenize(cleaned)
    stop_words = get_stopwords()
    return " ".join(
        lemmatize_token(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    )


def find_suspicious_words(text: str, keywords: tuple[str, ...]) -> list[str]:
    """Return suspicious keywords found in a text, preserving keyword order."""
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword in lowered]
