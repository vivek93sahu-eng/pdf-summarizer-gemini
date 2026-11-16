"""PDF Summarizer AI - a LangChain + Gemini document assistant."""

from .config import APP_NAME, VERSION, Settings
from .pdf_loader import PDFDocument, PDFLoadError, load_pdf
from .summarizer import PDFSummarizer, SummaryResult, SummarizationError

__all__ = [
    "APP_NAME",
    "VERSION",
    "Settings",
    "PDFDocument",
    "PDFLoadError",
    "load_pdf",
    "PDFSummarizer",
    "SummaryResult",
    "SummarizationError",
]
