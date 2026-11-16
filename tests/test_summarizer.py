"""Tests for chunking, summarisation modes and error handling."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.summarizer import PDFSummarizer, SummarizationError


def _patch_engine(monkeypatch, summarizer: PDFSummarizer, responses: list[str]) -> None:
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(summarizer, "_engine", lambda: fake)


def test_build_chunks_covers_every_page(settings, document):
    summarizer = PDFSummarizer(settings)
    chunks = summarizer.build_chunks(document)

    assert chunks
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == document.page_count
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.tokens > 0 for c in chunks)
    assert "Attention" in chunks[0].text


def test_build_chunks_splits_oversized_pages(settings):
    from src.pdf_loader import PDFDocument, Page

    text = "This section discusses attention mechanisms and their properties. " * 200
    big = PDFDocument(
        name="big.pdf",
        data=b"x",
        pages=[Page(number=1, text=text.strip(), char_count=len(text))],
        size_bytes=1,
        metadata={},
    )
    settings.chunk_tokens = 200
    settings.chunk_overlap_tokens = 20
    summarizer = PDFSummarizer(settings)

    chunks = summarizer.build_chunks(big)

    assert len(chunks) > 1
    assert all(c.tokens <= 200 for c in chunks)
    assert all(c.page_start == c.page_end == 1 for c in chunks)


def test_map_reduce_summary(settings, long_document, monkeypatch):
    settings.chunk_tokens = 200
    settings.chunk_overlap_tokens = 20
    summarizer = PDFSummarizer(settings)
    chunk_count = len(summarizer.build_chunks(long_document))
    assert chunk_count >= 3

    partials = [f"PART {index}" for index in range(1, chunk_count + 1)]
    _patch_engine(monkeypatch, summarizer, partials + ["FINAL SUMMARY"])

    result = summarizer.summarize(long_document, style="TL;DR", length="Brief")

    assert result.summary == "FINAL SUMMARY"
    assert result.chunk_count == chunk_count
    assert result.partial_summaries == partials
    assert result.mode == "map_reduce"
    assert result.elapsed_seconds >= 0


def test_refine_summary(settings, long_document, monkeypatch):
    settings.chunk_tokens = 200
    settings.chunk_overlap_tokens = 20
    summarizer = PDFSummarizer(settings)
    chunk_count = len(summarizer.build_chunks(long_document))
    assert chunk_count >= 3

    responses = [f"STEP {index}" for index in range(1, chunk_count + 1)]
    _patch_engine(monkeypatch, summarizer, responses)

    result = summarizer.summarize(long_document, style="Executive brief", mode="refine")

    assert result.summary == f"STEP {chunk_count}"
    assert len(result.partial_summaries) == chunk_count
    assert result.mode == "refine"


def test_single_chunk_uses_direct_prompt(settings, document, monkeypatch):
    summarizer = PDFSummarizer(settings)
    assert len(summarizer.build_chunks(document)) == 1
    _patch_engine(monkeypatch, summarizer, ["ONE SHOT"])

    result = summarizer.summarize(document)

    assert result.summary == "ONE SHOT"
    assert result.chunk_count == 1


def test_progress_callback_receives_updates(settings, long_document, monkeypatch):
    events: list[tuple[str, int, int]] = []

    def on_progress(stage: str, current: int, total: int, message: str) -> None:
        events.append((stage, current, total))

    settings.chunk_tokens = 200
    settings.chunk_overlap_tokens = 20
    summarizer = PDFSummarizer(settings, on_progress=on_progress)
    chunk_count = len(summarizer.build_chunks(long_document))
    _patch_engine(
        monkeypatch,
        summarizer,
        [f"P{index}" for index in range(chunk_count)] + ["FINAL"],
    )

    summarizer.summarize(long_document)

    assert events
    assert events[-1][0] == "Done"
    assert any(stage == "Mapping" for stage, _, _ in events)


def test_empty_document_raises(settings, document):
    document.pages[0].text = ""
    document.pages[1].text = ""
    document.pages[2].text = ""
    summarizer = PDFSummarizer(settings)

    with pytest.raises(SummarizationError):
        summarizer.summarize(document)


def test_missing_key_is_rejected():
    from src.config import ConfigError, Settings

    with pytest.raises(ConfigError):
        Settings(api_key="").validate()


def test_friendly_error_messages():
    assert "API key" in PDFSummarizer._friendly_error(Exception("401 Unauthenticated"))
    assert "rate limit" in PDFSummarizer._friendly_error(Exception("429 Resource exhausted"))
    assert "no longer available" in PDFSummarizer._friendly_error(
        Exception("404 models/gemini-2.5-flash is not found")
    )


def test_retryable_detection():
    assert PDFSummarizer._is_retryable("503 service unavailable")
    assert PDFSummarizer._is_retryable("rate limit exceeded")
    assert not PDFSummarizer._is_retryable("invalid argument")
