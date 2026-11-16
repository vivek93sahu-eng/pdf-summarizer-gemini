"""Small, dependency-light helpers shared across the app."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

_TOKEN_ENCODING_NAME = "o200k_base"


@lru_cache(maxsize=1)
def _encoder():
    """Return a cached tiktoken encoder, or ``None`` if it cannot be loaded."""
    try:
        import tiktoken

        return tiktoken.get_encoding(_TOKEN_ENCODING_NAME)
    except Exception:  # pragma: no cover - offline / wheel issues
        return None


def count_tokens(text: str) -> int:
    """Approximate token count using tiktoken, falling back to a heuristic."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim ``text`` so that it fits within ``max_tokens`` tokens."""
    if max_tokens <= 0:
        return ""
    enc = _encoder()
    if enc is None:
        return text[: max_tokens * 4]
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def message_text(content) -> str:
    """Normalise a LangChain message content into plain text.

    Gemini 3 models return content as a list of blocks (text, thought
    signatures, images) instead of a single string, so every call site needs
    this conversion.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                if block.get("type") in (None, "text") and block.get("text"):
                    parts.append(str(block["text"]))
                continue
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def response_text(response) -> str:
    """Extract text from an LangChain ``AIMessage``-like object."""
    return message_text(getattr(response, "content", response))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalise_whitespace(text: str) -> str:
    """Collapse noisy PDF whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def human_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs}s"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_filename(name: str) -> str:
    """Turn an uploaded file name into something safe for downloads."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "document"
    return re.sub(r"-{2,}", "-", stem)


@dataclass(slots=True)
class SourceStats:
    pages: int
    characters: int
    words: int

    @classmethod
    def from_pages(cls, pages: list[str]) -> "SourceStats":
        joined = "\n".join(pages)
        return cls(pages=len(pages), characters=len(joined), words=len(joined.split()))
