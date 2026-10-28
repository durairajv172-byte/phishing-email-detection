"""Custom feature extraction for phishing email detection."""
import re
import numpy as np
from scipy.sparse import csr_matrix

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
IP_RE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd")
KEYWORDS = ["urgent", "verify", "password", "account", "click", "suspended",
            "login", "bank", "confirm", "update", "winner", "free"]


def url_features(text):
    text = str(text)
    urls = URL_RE.findall(text)
    return [
        len(urls),                                              # number of URLs
        int(any(IP_RE.search(u) for u in urls)),                # IP-based URL
        int(any(len(u) > 75 for u in urls)),                    # very long URL
        int(any(s in u.lower() for u in urls for s in SHORTENERS)),  # shortened URL
        int("@" in text),                                       # '@' symbol
        int(any(u.lower().startswith("http://") for u in urls)),     # non-HTTPS link
    ]


def keyword_features(text):
    text = str(text).lower()
    return [text.count(k) for k in KEYWORDS]


def extra_features(texts):
    """Handcrafted URL + keyword features as a sparse matrix."""
    rows = [url_features(t) + keyword_features(t) for t in texts]
    return csr_matrix(np.array(rows, dtype=float))
