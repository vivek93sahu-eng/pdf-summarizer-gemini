"""PDF loading and text extraction built on top of ``pypdf``."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import Settings
from .utils import SourceStats, human_bytes, normalise_whitespace


class PDFLoadError(RuntimeError):
    """Raised with a user-friendly message when a PDF cannot be processed."""


@dataclass(slots=True)
class Page:
    number: int  # 1-based, matches what a reader sees in a PDF viewer
    text: str
    char_count: int

    @property
    def is_empty(self) -> bool:
        return self.char_count == 0


@dataclass(slots=True)
class PDFDocument:
    name: str
    data: bytes
    pages: list[Page]
    size_bytes: int
    metadata: dict

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """Full document text with page markers."""
        return "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in self.pages if p.text)

    @property
    def plain_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def stats(self) -> SourceStats:
        return SourceStats.from_pages([p.text for p in self.pages])

    @property
    def empty_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.is_empty]

    @property
    def looks_scanned(self) -> bool:
        """True when most pages contain little or no extractable text."""
        if not self.pages:
            return True
        thin = [p for p in self.pages if p.char_count < 40]
        return len(thin) / len(self.pages) >= 0.6

    @property
    def title(self) -> str:
        raw = (self.metadata or {}).get("/Title")
        if isinstance(raw, str) and raw.strip() and raw.strip().lower() != "untitled":
            return raw.strip()
        return Path(self.name).stem

    def pages_text(self, numbers: list[int] | None = None) -> str:
        selected = self.pages if numbers is None else [p for p in self.pages if p.number in set(numbers)]
        return "\n\n".join(p.text for p in selected if p.text)


def load_pdf(source: bytes | bytearray | Path | str, name: str | None = None, settings: Settings | None = None) -> PDFDocument:
    """Read a PDF from raw bytes or a filesystem path.

    ``settings`` is only used for the guard rails (file size / page count).
    """
    cfg = settings or Settings.from_env()

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise PDFLoadError(f"File not found: {path}")
        data = path.read_bytes()
        filename = name or path.name
    else:
        data = bytes(source)
        filename = name or "document.pdf"

    if not data:
        raise PDFLoadError("The uploaded file is empty.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > cfg.max_file_mb:
        raise PDFLoadError(
            f"`{filename}` is {human_bytes(len(data))}, which exceeds the "
            f"{cfg.max_file_mb} MB limit. Try compressing the PDF or splitting it."
        )

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            # Many PDFs are encrypted with an empty user password; try that first.
            try:
                if reader.decrypt("") == 0:
                    raise PDFLoadError(
                        f"`{filename}` is password protected. Remove the password "
                        "and upload it again."
                    )
            except PDFLoadError:
                raise
            except Exception as exc:  # pragma: no cover - pypdf edge cases
                raise PDFLoadError(
                    f"`{filename}` is encrypted and could not be opened."
                ) from exc

        page_total = len(reader.pages)
        if page_total == 0:
            raise PDFLoadError(f"`{filename}` contains no pages.")
        if page_total > cfg.max_pages:
            raise PDFLoadError(
                f"`{filename}` has {page_total} pages, above the {cfg.max_pages}-page "
                "limit.",
            )

        pages: list[Page] = []
        for index, raw_page in enumerate(reader.pages):
            try:
                extracted = raw_page.extract_text() or ""
            except Exception:  # pragma: no cover - malformed content streams
                extracted = ""
            cleaned = normalise_whitespace(extracted)
            pages.append(Page(number=index + 1, text=cleaned, char_count=len(cleaned)))

        metadata: dict = {}
        try:
            metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items()}
        except Exception:  # pragma: no cover
            metadata = {}

    except PDFLoadError:
        raise
    except PdfReadError as exc:
        raise PDFLoadError(
            f"`{filename}` does not look like a valid PDF ({exc})."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise PDFLoadError(f"Could not read `{filename}`: {exc}") from exc

    document = PDFDocument(
        name=filename,
        data=data,
        pages=pages,
        size_bytes=len(data),
        metadata=metadata,
    )

    if not document.plain_text.strip():
        raise PDFLoadError(
            f"No text could be extracted from `{filename}`. It is probably a "
            "scanned document; run it through OCR first, then try again."
        )

    return document
