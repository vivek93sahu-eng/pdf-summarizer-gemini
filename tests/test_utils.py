"""Unit tests for the shared helpers."""

from __future__ import annotations

import pytest

from src.utils import (
    count_tokens,
    human_bytes,
    human_duration,
    message_text,
    normalise_whitespace,
    safe_filename,
    sha256_bytes,
    truncate_to_tokens,
)


def test_normalise_whitespace_collapses_noise():
    raw = "Hello   world\r\n\r\n\r\nSecond   line\t here  "
    assert normalise_whitespace(raw) == "Hello world\n\nSecond line here"


def test_count_tokens_and_truncate():
    text = "word " * 500
    tokens = count_tokens(text)
    assert tokens > 100
    trimmed = truncate_to_tokens(text, 20)
    assert 0 < count_tokens(trimmed) <= 21


def test_truncate_to_zero_tokens():
    assert truncate_to_tokens("anything", 0) == ""


def test_sha256_is_stable():
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
    assert sha256_bytes(b"abc") != sha256_bytes(b"abd")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB")],
)
def test_human_bytes(value, expected):
    assert human_bytes(value) == expected


def test_human_duration():
    assert human_duration(0.25).endswith("ms")
    assert human_duration(12.34) == "12.3 s"
    assert human_duration(95) == "1m 35s"


def test_safe_filename_strips_unsafe_characters():
    assert safe_filename("my paper (final)?.pdf") == "my-paper-final-.pdf"
    assert safe_filename("   ") == "document"


def test_message_text_handles_gemini_content_blocks():
    blocks = [
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
        {"type": "thought", "text": "ignored"},
    ]
    assert message_text(blocks) == "Hello world"
    assert message_text("plain") == "plain"
    assert message_text(None) == ""
