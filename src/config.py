"""Central configuration for the PDF Summarizer AI application.

Everything that can be tuned lives here so the UI layer stays thin. Values are
read from environment variables (optionally from a local ``.env`` file) with
sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env once, without overriding variables that are already exported.
load_dotenv(PROJECT_ROOT / ".env", override=False)

APP_NAME = "PDF Summarizer AI"
APP_TAGLINE = "Turn long documents into clear answers with LangChain + Gemini"
APP_ICON = "📄"
VERSION = "1.0.0"

DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Models exposed in the sidebar. The first entry is the default and is the one
# exercised by the smoke test. Google retires preview models regularly, so the
# list stays short and easy to edit.
MODEL_CHOICES: dict[str, str] = {
    "Gemini 3.1 Flash Lite (fast, default)": "gemini-3.1-flash-lite",
    "Gemini 3.5 Flash (balanced)": "gemini-3.5-flash",
    "Gemini 3.5 Flash Lite (cheapest)": "gemini-3.5-flash-lite",
    "Gemini 3.1 Pro (highest quality)": "gemini-3.1-pro-preview",
}

# Fallback chain used when the primary model is temporarily unavailable.
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-flash-lite-latest"]

STYLE_INSTRUCTIONS: dict[str, str] = {
    "Executive brief": (
        "Write a polished executive summary for a busy decision maker. Lead with "
        "the single most important takeaway, then cover the key findings, and end "
        "with implications or recommended next steps."
    ),
    "Key points (bullets)": (
        "Summarise the document as a set of scannable bullet points. Group related "
        "points under short bold headings when the document has natural sections. "
        "Every bullet must be a complete, self-contained statement."
    ),
    "TL;DR": (
        "Write an extremely concise TL;DR of at most three sentences. No preamble, "
        "no headings, no bullet points - just the essence of the document."
    ),
    "Narrative summary": (
        "Write a flowing, well-structured narrative summary in 3-5 paragraphs. "
        "Preserve the author's line of reasoning and connect ideas with clear "
        "transitions rather than listing facts."
    ),
    "Study notes": (
        "Produce structured study notes: an overview, the core concepts with short "
        "definitions, important facts or figures, and a short 'questions to test "
        "yourself' section with 3-5 questions."
    ),
}

LENGTH_INSTRUCTIONS: dict[str, str] = {
    "Brief": "Keep the final output under roughly 200 words.",
    "Standard": "Aim for roughly 350-500 words.",
    "In-depth": "Aim for 700-1000 words, with enough detail to stand alone.",
}

LANGUAGE_CHOICES = [
    "Same as document",
    "English",
    "Hindi",
    "Spanish",
    "French",
    "German",
    "Portuguese",
    "Japanese",
]

MODES = {
    "Map-reduce (fast, parallel chunks)": "map_reduce",
    "Refine (slower, best coherence)": "refine",
}


class ConfigError(RuntimeError):
    """Raised when the application is misconfigured (e.g. missing API key)."""


@dataclass(slots=True)
class Settings:
    """Runtime configuration resolved from the environment."""

    api_key: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_output_tokens: int = 4096

    # Chunking (measured in tokens of the o200k_base tokenizer).
    chunk_tokens: int = 6000
    chunk_overlap_tokens: int = 400

    # Guard rails for uploaded files.
    max_file_mb: int = 50
    max_pages: int = 500

    # Networking.
    request_timeout: int = 180
    max_retries: int = 3

    # Text used when a document has no extractable text at all.
    scanned_page_char_threshold: int = 40
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
            or ""
        ).strip()
        model = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()

        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, default))
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            api_key=api_key,
            model=model,
            temperature=_float("GEMINI_TEMPERATURE", 0.2),
            max_output_tokens=_int("GEMINI_MAX_OUTPUT_TOKENS", 4096),
            chunk_tokens=_int("CHUNK_TOKENS", 6000),
            chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", 400),
            max_file_mb=_int("MAX_FILE_MB", 50),
            max_pages=_int("MAX_PAGES", 500),
            request_timeout=_int("REQUEST_TIMEOUT", 180),
            max_retries=_int("MAX_RETRIES", 3),
        )

    def validate(self) -> None:
        """Raise :class:`ConfigError` when the settings cannot work."""
        if not self.api_key:
            raise ConfigError(
                "No Gemini API key found. Create a `.env` file in the project root "
                "with `GEMINI_API_KEY=...`, or export the variable in your shell. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        if len(self.api_key) < 20:
            raise ConfigError(
                "GEMINI_API_KEY looks too short to be valid. Double-check that you "
                "pasted the whole key."
            )
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ConfigError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TOKENS.")
        if self.chunk_tokens < 200:
            raise ConfigError("CHUNK_TOKENS is too small; use at least 200.")

    @property
    def masked_key(self) -> str:
        if not self.api_key:
            return "not set"
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"
