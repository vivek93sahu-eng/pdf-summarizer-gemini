"""Tests for the PDF loader and retrieval-based Q&A."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.pdf_loader import PDFLoadError, PDFDocument, Page, load_pdf
from src.qa import DocumentChat

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "sample-1page.pdf"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")
def test_load_real_pdf(settings):
    document = load_pdf(SAMPLE, settings=settings)

    assert document.page_count >= 1
    assert document.plain_text
    assert document.stats.words > 0


def test_load_rejects_garbage(settings):
    with pytest.raises(PDFLoadError):
        load_pdf(b"this is definitely not a pdf", settings=settings)


def test_load_rejects_empty(settings):
    with pytest.raises(PDFLoadError):
        load_pdf(b"", settings=settings)


def test_missing_file(settings):
    with pytest.raises(PDFLoadError):
        load_pdf(ROOT / "does-not-exist.pdf", settings=settings)


def test_document_helpers(document):
    assert document.page_count == 3
    assert document.empty_pages == []
    assert not document.looks_scanned
    assert document.title == "Attention Is All You Need"
    assert "Page 2" in document.text
    assert "encoder" in document.pages_text([2])
    assert "encoder" not in document.pages_text([3])


def test_scanned_detection():
    document = PDFDocument(
        name="scan.pdf",
        data=b"x",
        pages=[Page(number=i, text="", char_count=0) for i in range(1, 5)],
        size_bytes=1,
        metadata={},
    )
    assert document.looks_scanned


def test_retrieval_prefers_matching_page(document, settings):
    chat = DocumentChat(document, settings)

    passages = chat._retrieve("What BLEU score did they report?")

    assert passages
    assert passages[0].page == 3


def test_retrieval_handles_no_matches(document, settings):
    chat = DocumentChat(document, settings)

    passages = chat._retrieve("zzzzz qqqqq")

    assert passages  # falls back to the first passages instead of nothing


def test_chat_answer_cites_pages(document, settings, monkeypatch):
    chat = DocumentChat(document, settings)
    monkeypatch.setattr(
        chat, "_engine", lambda: FakeListChatModel(responses=["They reached 28.4 BLEU (p. 3)."])
    )

    answer, sources = chat.answer("What BLEU score did they report?")

    assert "28.4" in answer
    assert 3 in sources


def test_chat_rejects_empty_question(document, settings):
    chat = DocumentChat(document, settings)
    answer, sources = chat.answer("   ")
    assert "Ask a question" in answer
    assert sources == []


def test_suggested_questions(document, settings):
    chat = DocumentChat(document, settings)
    questions = chat.suggested_questions()
    assert 2 <= len(questions) <= 4
    assert all(isinstance(q, str) and q for q in questions)
