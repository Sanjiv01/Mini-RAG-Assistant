"""Fetch documents from URLs and return their plain-text content.

Supports two content types:
  * PDF (application/pdf or .pdf URL)  → parsed via pypdf
  * HTML (text/html or unspecified)    → parsed via BeautifulSoup, scripts/
                                          nav/footer stripped

Stdlib networking; no requests/httpx dependency.
"""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Tuple

# Reject pages larger than this many chars (after extraction).
_MAX_CHARS = 500_000
_TIMEOUT_S = 30
_USER_AGENT = "Hermes-RAG/1.0 (+https://github.com)"


def fetch_url(url: str) -> Tuple[bytes, str]:
    """Download URL with a sane timeout and return (bytes, content_type)."""
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read()
            return data, content_type
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e.reason}") from e


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(pages)


def _extract_html(data: bytes) -> Tuple[str, str]:
    """Return (text, page_title)."""
    from bs4 import BeautifulSoup

    # Try to decode safely.
    text = data.decode("utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")

    # Drop boilerplate.
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "header", "aside", "form", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Prefer <article> / <main> if present; fall back to <body>.
    root = soup.find("article") or soup.find("main") or soup.body or soup

    body_text = root.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines.
    body_text = re.sub(r"\n{3,}", "\n\n", body_text)
    return body_text, title


def _slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9._\-]", "", text)
    return text[:max_len].strip("-._") or "page"


def load_url_as_text(url: str) -> Tuple[str, str]:
    """Fetch a URL and return (text, display_name).

    Raises RuntimeError on network failure or unsupported content.
    """
    data, content_type = fetch_url(url)
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()

    is_pdf = "pdf" in content_type or path.endswith(".pdf")
    is_html = (
        "html" in content_type
        or "xml" in content_type
        or content_type == ""
        or path.endswith((".html", ".htm", "/"))
        or "." not in path.rsplit("/", 1)[-1]
    )

    if is_pdf:
        text = _extract_pdf(data)
        if not text.strip():
            raise RuntimeError("PDF parsed but no extractable text (scanned image?).")
        # Display name from path or fallback.
        leaf = path.rsplit("/", 1)[-1] or "document.pdf"
        if not leaf.endswith(".pdf"):
            leaf += ".pdf"
        return text[:_MAX_CHARS], leaf

    if is_html:
        text, title = _extract_html(data)
        if not text.strip():
            raise RuntimeError(
                "Couldn't extract readable text from this page. It may be "
                "JavaScript-rendered or behind a login."
            )
        display = (title or parsed.netloc or "page")
        return text[:_MAX_CHARS], _slug(display) + ".html"

    raise RuntimeError(
        f"Unsupported content type: {content_type or 'unknown'}. "
        "Only PDF and HTML URLs are supported."
    )
