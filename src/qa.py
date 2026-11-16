"""Retrieval-augmented Q&A over a single PDF document.

The app intentionally avoids a vector database: a compact BM25 index over page
chunks runs instantly, needs no extra services, and keeps the project easy to
run. Swapping in embeddings later only requires replacing :meth:`_retrieve`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings
from .pdf_loader import PDFDocument
from .prompts import QA_PROMPT
from .utils import count_tokens, response_text, truncate_to_tokens

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]+")

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did",
    "does", "what", "when", "where", "which", "while", "with", "this", "that",
    "these", "those", "from", "have", "into", "than", "then", "them", "they",
    "there", "their", "about", "would", "could", "should", "also", "such", "been",
    "being", "were", "will", "more", "most", "some", "only", "over", "very",
    "your", "just", "like", "make", "made", "many", "much", "each", "other",
    "document", "pdf",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


@dataclass(slots=True)
class Passage:
    text: str
    page: int
    tokens: int
    tf: Counter = field(default_factory=Counter)


class DocumentChat:
    """Answer questions about one document, with page citations."""

    def __init__(
        self,
        document: PDFDocument,
        settings: Settings,
        context_tokens: int = 18_000,
        top_k: int = 6,
    ):
        self.document = document
        self.settings = settings
        self.context_tokens = context_tokens
        self.top_k = top_k
        self.passages = self._build_passages()
        self._idf = self._build_idf()
        self._avg_len = (
            sum(p.tokens for p in self.passages) / len(self.passages)
            if self.passages
            else 0.0
        )
        self._llm: ChatGoogleGenerativeAI | None = None

    # ------------------------------------------------------------------ public
    def answer(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> tuple[str, list[int]]:
        if not question.strip():
            return "Ask a question about the document to get started.", []

        selected = self._retrieve(question)
        if not selected:
            return (
                "I could not find anything relevant in this document. Try rephrasing "
                "the question with terms that appear in the text.",
                [],
            )

        context_parts: list[str] = []
        used = 0
        sources: list[int] = []
        budget = self.context_tokens
        for passage in selected:
            block = f"(page {passage.page})\n{passage.text}"
            block_tokens = count_tokens(block)
            if used + block_tokens > budget:
                block = truncate_to_tokens(block, max(0, budget - used))
                block_tokens = count_tokens(block)
            if block_tokens <= 0:
                break
            context_parts.append(block)
            used += block_tokens
            if passage.page not in sources:
                sources.append(passage.page)

        history_text = "None."
        if history:
            recent = history[-4:]
            history_text = "\n".join(
                f"{'User' if role == 'user' else 'Assistant'}: {text[:700]}"
                for role, text in recent
            )

        messages = QA_PROMPT.format_messages(
            title=self.document.title,
            context="\n\n---\n\n".join(context_parts),
            history=history_text,
            question=question,
        )
        response = self._engine().invoke(messages)
        return response_text(response), sorted(sources)

    def suggested_questions(self, limit: int = 4) -> list[str]:
        """Cheap, heuristic starter questions based on the document's own words."""
        counts: Counter[str] = Counter()
        for passage in self.passages[:20]:
            counts.update(_tokens(passage.text)[:400])

        generic = {"figure", "table", "section", "chapter", "figure", "page", "et", "al"}
        keywords = [w for w, _ in counts.most_common(40) if w not in generic and len(w) > 3][:6]

        questions = ["What is this document about?", "What are the key takeaways?"]
        if len(keywords) >= 2:
            questions.append(f"How does the document treat '{keywords[0]}' and '{keywords[1]}'?")
        if len(keywords) >= 4:
            questions.append(f"What does it say about '{keywords[2]}'?")
        return questions[:limit]

    # ----------------------------------------------------------------- private
    def _build_passages(self) -> list[Passage]:
        passages: list[Passage] = []
        for page in self.document.pages:
            text = page.text.strip()
            if not text:
                continue
            # Keep passages around 900 tokens so retrieval stays selective.
            words = text.split()
            step = 600
            for start in range(0, len(words), step):
                block = " ".join(words[start : start + step * 2]).strip()
                if not block:
                    continue
                passages.append(
                    Passage(
                        text=block,
                        page=page.number,
                        tokens=len(_tokens(block)),
                        tf=Counter(_tokens(block)),
                    )
                )
                if start + step * 2 >= len(words):
                    break
        return passages

    def _build_idf(self) -> dict[str, float]:
        total = len(self.passages)
        df: Counter[str] = Counter()
        for passage in self.passages:
            df.update(set(passage.tf))
        return {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def _retrieve(self, question: str) -> list[Passage]:
        query_terms = _tokens(question)
        if not query_terms or not self.passages:
            return self.passages[: self.top_k]

        k1, b = 1.5, 0.75
        scored: list[tuple[float, Passage]] = []
        for passage in self.passages:
            score = 0.0
            for term in query_terms:
                freq = passage.tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + k1 * (1 - b + b * (passage.tokens / max(self._avg_len, 1)))
                score += idf * (freq * (k1 + 1)) / denom
            if score > 0:
                scored.append((score, passage))

        if not scored:  # fall back to the first pages when nothing matches
            return self.passages[: self.top_k]

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [passage for _, passage in scored[: self.top_k]]

    def _engine(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.settings.model,
                google_api_key=self.settings.api_key,
                temperature=0.2,
                max_output_tokens=self.settings.max_output_tokens,
                timeout=self.settings.request_timeout,
                max_retries=2,
            )
        return self._llm

    # Convenience helpers for anyone who wants LangChain message objects.
    @staticmethod
    def to_lc_history(history: list[tuple[str, str]]):
        messages = []
        for role, text in history:
            messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
        return messages

    @staticmethod
    def system_message() -> SystemMessage:
        return SystemMessage(content="You answer questions about a PDF document.")
