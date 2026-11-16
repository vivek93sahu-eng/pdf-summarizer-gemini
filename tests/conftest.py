"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.pdf_loader import PDFDocument, Page


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        api_key="test-key-" + "x" * 32,
        model="gemini-3.1-flash-lite",
        chunk_tokens=800,
        chunk_overlap_tokens=100,
        max_retries=1,
    )


@pytest.fixture()
def document() -> PDFDocument:
    pages = [
        Page(
            number=1,
            text=(
                "Attention Is All You Need. The dominant sequence transduction models "
                "are based on complex recurrent or convolutional neural networks. We "
                "propose a new simple network architecture, the Transformer, based "
                "solely on attention mechanisms. Experiments show superior quality."
            ),
            char_count=270,
        ),
        Page(
            number=2,
            text=(
                "The encoder is composed of a stack of six identical layers. Each layer "
                "has two sub-layers: multi-head self-attention and a position-wise "
                "feed-forward network. Residual connections surround each sub-layer "
                "followed by layer normalisation."
            ),
            char_count=252,
        ),
        Page(
            number=3,
            text=(
                "We trained on the WMT 2014 English-to-German translation task and "
                "achieved a BLEU score of 28.4, improving over the previous best "
                "results by more than 2 BLEU. Training took 3.5 days on eight GPUs."
            ),
            char_count=203,
        ),
    ]
    return PDFDocument(
        name="attention.pdf",
        data=b"%PDF-1.4 fake",
        pages=pages,
        size_bytes=13,
        metadata={"/Title": "Attention Is All You Need"},
    )


@pytest.fixture()
def long_document(document) -> PDFDocument:
    """A ~600-token document, long enough to require several chunks."""
    pages: list[Page] = []
    for _ in range(4):
        for page in document.pages:
            pages.append(
                Page(
                    number=len(pages) + 1,
                    text=page.text,
                    char_count=page.char_count,
                )
            )
    return PDFDocument(
        name="long-attention.pdf",
        data=b"%PDF-1.4 fake",
        pages=pages,
        size_bytes=13,
        metadata={"/Title": "Attention Is All You Need (long)"},
    )
