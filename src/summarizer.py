"""Chunking + LangChain summarisation chains for PDF documents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import FALLBACK_MODELS, Settings
from .pdf_loader import PDFDocument
from .prompts import (
    DIRECT_PROMPT,
    MAP_PROMPT,
    REDUCE_PROMPT,
    REFINE_PROMPT,
    length_instruction,
    language_instruction,
    style_instruction,
)
from .utils import count_tokens, response_text, truncate_to_tokens

ProgressCallback = Callable[[str, int, int, str], None]


class SummarizationError(RuntimeError):
    """Raised when the model cannot produce a summary."""


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    tokens: int


@dataclass(slots=True)
class SummaryResult:
    summary: str
    document_name: str
    mode: str
    style: str
    length: str
    language: str
    model: str
    chunk_count: int
    partial_summaries: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)


class PDFSummarizer:
    """Summarise a :class:`PDFDocument` with Gemini through LangChain."""

    def __init__(self, settings: Settings, on_progress: ProgressCallback | None = None):
        settings.validate()
        self.settings = settings
        self.on_progress = on_progress
        self._llm: ChatGoogleGenerativeAI | None = None
        self._active_model = settings.model

    # ------------------------------------------------------------------ public
    def build_chunks(self, document: PDFDocument) -> list[Chunk]:
        """Split the document into overlapping, page-aware chunks."""
        cfg = self.settings
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="o200k_base",
            chunk_size=cfg.chunk_tokens,
            chunk_overlap=cfg.chunk_overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_pages: list[int] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_pages, buffer_tokens
            if not buffer:
                return
            text = "\n\n".join(buffer).strip()
            if text:
                chunks.append(
                    Chunk(
                        index=len(chunks) + 1,
                        text=text,
                        page_start=buffer_pages[0],
                        page_end=buffer_pages[-1],
                        tokens=count_tokens(text),
                    )
                )
            buffer, buffer_pages, buffer_tokens = [], [], 0

        for page in document.pages:
            if not page.text:
                continue
            page_tokens = count_tokens(page.text)

            # A single oversized page gets split on its own.
            if page_tokens > cfg.chunk_tokens:
                flush()
                for piece in splitter.split_text(page.text):
                    if not piece.strip():
                        continue
                    chunks.append(
                        Chunk(
                            index=len(chunks) + 1,
                            text=piece.strip(),
                            page_start=page.number,
                            page_end=page.number,
                            tokens=count_tokens(piece),
                        )
                    )
                continue

            if buffer_tokens + page_tokens > cfg.chunk_tokens and buffer:
                flush()

            buffer.append(page.text)
            buffer_pages.append(page.number)
            buffer_tokens += page_tokens

        flush()
        return chunks

    def summarize(
        self,
        document: PDFDocument,
        style: str = "Executive brief",
        length: str = "Standard",
        mode: str = "map_reduce",
        language: str = "Same as document",
    ) -> SummaryResult:
        started = time.perf_counter()
        chunks = self.build_chunks(document)
        if not chunks:
            raise SummarizationError(
                "There is no extractable text to summarise in this document."
            )

        usage = _Usage()
        style_i = style_instruction(style)
        length_i = length_instruction(length)
        language_i = language_instruction(language)
        partials: list[str] = []

        total_steps = len(chunks) if mode == "refine" else len(chunks) + 2
        self._report("Preparing", 0, total_steps, f"{len(chunks)} chunk(s) ready")

        if len(chunks) == 1:
            text = truncate_to_tokens(chunks[0].text, self.settings.chunk_tokens)
            messages = DIRECT_PROMPT.format_messages(
                title=document.title,
                style_instruction=style_i,
                length_instruction=length_i,
                language_instruction=language_i,
                text=text,
            )
            self._report("Summarising", 1, 3, "Reading the document")
            summary = self._generate(messages, usage)
            self._report("Done", 3, 3, "Summary ready")
        elif mode == "refine":
            partials, summary = self._summarize_refine(
                document, chunks, style_i, length_i, language_i, usage
            )
        else:
            partials, summary = self._summarize_map_reduce(
                document, chunks, style_i, length_i, language_i, usage
            )

        elapsed = time.perf_counter() - started
        from .utils import utc_timestamp

        return SummaryResult(
            summary=summary.strip(),
            document_name=document.name,
            mode=mode,
            style=style,
            length=length,
            language=language,
            model=self._active_model,
            chunk_count=len(chunks),
            partial_summaries=partials,
            elapsed_seconds=elapsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            created_at=utc_timestamp(),
        )

    def summarize_page(self, document: PDFDocument, page_number: int) -> str:
        page = next((p for p in document.pages if p.number == page_number), None)
        if page is None or not page.text:
            return "(no extractable text)"
        usage = _Usage()
        messages = PAGE_PROMPT.format_messages(
            page=page.number, title=document.title, text=page.text
        )
        return self._generate(messages, usage).strip()

    # ----------------------------------------------------------------- private
    def _summarize_map_reduce(
        self,
        document: PDFDocument,
        chunks: Sequence[Chunk],
        style_i: str,
        length_i: str,
        language_i: str,
        usage: "_Usage",
    ) -> tuple[list[str], str]:
        partials: list[str] = []
        total = len(chunks) + 2

        for position, chunk in enumerate(chunks, start=1):
            self._report(
                "Mapping",
                position,
                total,
                f"Summarising chunk {position} of {len(chunks)} (page {chunk.page_start}-{chunk.page_end})",
            )
            messages = MAP_PROMPT.format_messages(
                index=position,
                total=len(chunks),
                page=chunk.page_start,
                text=truncate_to_tokens(chunk.text, self.settings.chunk_tokens),
            )
            partials.append(self._generate(messages, usage).strip())

        self._report("Reducing", len(chunks) + 1, total, "Combining partial summaries")
        summary = self._reduce(document, partials, style_i, length_i, language_i, usage)
        self._report("Done", total, total, "Summary ready")
        return partials, summary

    def _reduce(
        self,
        document: PDFDocument,
        partials: Iterable[str],
        style_i: str,
        length_i: str,
        language_i: str,
        usage: "_Usage",
    ) -> str:
        """Reduce partial summaries, batching them when they are too long."""
        items = [p for p in partials if p.strip()]
        if not items:
            raise SummarizationError("The model returned no partial summaries.")
        if len(items) == 1:
            return items[0]

        budget = int(self.settings.chunk_tokens * 0.8)
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for item in items:
            item_tokens = count_tokens(item)
            if current and current_tokens + item_tokens > budget:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(item)
            current_tokens += item_tokens
        if current:
            batches.append(current)

        if len(batches) == 1:
            combined = "\n\n".join(batches[0])
            messages = REDUCE_PROMPT.format_messages(
                title=document.title,
                style_instruction=style_i,
                length_instruction=length_i,
                language_instruction=language_i,
                summaries=combined,
            )
            return self._generate(messages, usage).strip()

        # Too much text: summarise each batch first, then reduce the results.
        intermediates = [self._reduce(document, batch, style_i, length_i, language_i, usage) for batch in batches]
        return self._reduce(document, intermediates, style_i, length_i, language_i, usage)

    def _summarize_refine(
        self,
        document: PDFDocument,
        chunks: Sequence[Chunk],
        style_i: str,
        length_i: str,
        language_i: str,
        usage: "_Usage",
    ) -> tuple[list[str], str]:
        partials: list[str] = []
        total = len(chunks)
        summary = ""

        for position, chunk in enumerate(chunks, start=1):
            if position == 1:
                self._report(
                    "Reading",
                    position,
                    total,
                    f"Drafting from chunk 1 of {total} (page {chunk.page_start})",
                )
                messages = MAP_PROMPT.format_messages(
                    index=1,
                    total=total,
                    page=chunk.page_start,
                    text=truncate_to_tokens(chunk.text, self.settings.chunk_tokens),
                )
                summary = self._generate(messages, usage).strip()
            else:
                self._report(
                    "Refining",
                    position,
                    total,
                    f"Refining with chunk {position} of {total} (page {chunk.page_start})",
                )
                messages = REFINE_PROMPT.format_messages(
                    title=document.title,
                    summary=truncate_to_tokens(summary, self.settings.chunk_tokens // 2),
                    page=chunk.page_start,
                    text=truncate_to_tokens(chunk.text, self.settings.chunk_tokens),
                    style_instruction=style_i,
                    length_instruction=length_i,
                    language_instruction=language_i,
                )
                summary = self._generate(messages, usage).strip()
            partials.append(summary)

        self._report("Done", total, total, "Summary ready")
        return partials, summary

    def _engine(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self._active_model,
                google_api_key=self.settings.api_key,
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
                timeout=self.settings.request_timeout,
                max_retries=0,  # retries are handled below so we can report them
            )
        return self._llm

    def _generate(self, messages: list[BaseMessage], usage: "_Usage") -> str:
        """Invoke the model with retries, quota handling and model fallback."""
        attempts = max(1, self.settings.max_retries)
        candidates = [self._active_model] + [
            m for m in FALLBACK_MODELS if m != self._active_model
        ]
        last_error: Exception | None = None

        for candidate in candidates:
            if candidate != self._active_model:
                self._active_model = candidate
                self._llm = None
            for attempt in range(1, attempts + 1):
                try:
                    response = self._engine().invoke(messages)
                    usage.add(response)
                    text = response_text(response)
                    if not text:
                        raise SummarizationError(
                            "Gemini returned an empty response. Try a different model or retry."
                        )
                    return text
                except SummarizationError:
                    raise
                except Exception as exc:  # noqa: BLE001 - normalised below
                    last_error = exc
                    message = str(exc)
                    lowered = message.lower()

                    if self._is_model_missing(lowered):
                        break  # try the next model immediately
                    if not self._is_retryable(lowered) or attempt == attempts:
                        raise SummarizationError(self._friendly_error(exc)) from exc

                    # Exponential backoff: 2s, 4s, 8s ...
                    time.sleep(min(2**attempt, 8))

        raise SummarizationError(self._friendly_error(last_error))

    @staticmethod
    def _is_model_missing(lowered: str) -> bool:
        return (
            "no longer available" in lowered
            or "is not found" in lowered
            or "not_found" in lowered
            or ("404" in lowered and "model" in lowered)
        )

    @staticmethod
    def _is_retryable(lowered: str) -> bool:
        return any(
            token in lowered
            for token in (
                "429",
                "resource exhausted",
                "resource_exhausted",
                "quota",
                "rate limit",
                "503",
                "unavailable",
                "500",
                "internal error",
                "deadline exceeded",
                "timeout",
                "temporarily",
            )
        )

    @staticmethod
    def _friendly_error(exc: Exception | None) -> str:
        message = str(exc) if exc else "Unknown error"
        lowered = message.lower()
        if "api key" in lowered or "unauthenticated" in lowered or "401" in lowered or "403" in lowered:
            return (
                "Gemini rejected the API key. Check that GEMINI_API_KEY in `.env` is "
                "correct and that the Gemini API is enabled for the project."
            )
        if "quota" in lowered or "429" in lowered or "resource exhausted" in lowered:
            return (
                "Gemini rate limit or quota reached. Wait a minute and retry, or "
                "switch to a lighter model in the sidebar."
            )
        if "safety" in lowered or "blocked" in lowered:
            return (
                "Gemini blocked this content for safety reasons. Try another style "
                "or a different document."
            )
        if "no longer available" in lowered or "404" in lowered:
            return (
                "The selected model is no longer available to your API key. Pick a "
                "different model in the sidebar and retry."
            )
        return f"Gemini request failed: {message}"

    def _report(self, stage: str, current: int, total: int, message: str) -> None:
        if self.on_progress:
            try:
                self.on_progress(stage, current, total, message)
            except Exception:  # pragma: no cover - UI callbacks must never break work
                pass
