"""Feature extraction for URL phishing detection."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qs, urlparse

import pandas as pd

from config import SUSPICIOUS_KEYWORDS


IP_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def shannon_entropy(value: str) -> float:
    """Compute Shannon entropy for a string."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def contains_ip(hostname: str) -> int:
    """Return 1 when the hostname is an IP address."""
    hostname = hostname.strip("[]")
    if not IP_PATTERN.match(hostname):
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_url_features(url: object) -> dict[str, float]:
    """Extract deterministic phishing indicators from a URL."""
    raw_url = "" if url is None else str(url).strip()
    parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    subdomains = [part for part in hostname.split(".")[:-2] if part]
    keyword_hits = sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in raw_url.lower())
    return {
        "url_length": float(len(raw_url)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "digit_count": float(sum(char.isdigit() for char in raw_url)),
        "dot_count": float(raw_url.count(".")),
        "slash_count": float(raw_url.count("/")),
        "subdomain_count": float(len(subdomains)),
        "query_param_count": float(len(parse_qs(query))),
        "contains_at": float("@" in raw_url),
        "contains_dash": float("-" in raw_url),
        "contains_percent": float("%" in raw_url),
        "contains_equal": float("=" in raw_url),
        "contains_ip": float(contains_ip(hostname)),
        "uses_http": float(parsed.scheme == "http"),
        "uses_https": float(parsed.scheme == "https"),
        "suspicious_keyword_count": float(keyword_hits),
        "entropy": shannon_entropy(raw_url),
    }


def feature_names() -> list[str]:
    """Return stable URL feature names."""
    return list(extract_url_features("https://example.com").keys())


def build_url_feature_frame(urls: pd.Series | list[str]) -> pd.DataFrame:
    """Build a feature matrix from URL strings."""
    return pd.DataFrame([extract_url_features(url) for url in urls], columns=feature_names())
