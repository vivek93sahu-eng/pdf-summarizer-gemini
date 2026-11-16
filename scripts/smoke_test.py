"""End-to-end connectivity check for the PDF Summarizer AI project.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_test.py

Checks, in order:
  1. `.env` is readable and a key is present
  2. LangChain can build a Gemini chat model
  3. Gemini answers a real request
  4. (optional) a sample PDF can be loaded and summarised

Exit code 0 means the project is ready to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

from src.config import Settings  # noqa: E402
from src.utils import response_text  # noqa: E402


def main() -> int:
    print("=" * 62)
    print("PDF Summarizer AI - smoke test")
    print("=" * 62)

    settings = Settings.from_env()
    print(f"[1/4] API key detected : {settings.masked_key}")
    try:
        settings.validate()
    except Exception as exc:
        print(f"      FAILED: {exc}")
        return 1
    print(f"      model configured : {settings.model}")

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model,
            google_api_key=settings.api_key,
            temperature=0,
            max_output_tokens=32,
            timeout=settings.request_timeout,
        )
    except Exception as exc:
        print(f"[2/4] FAILED to build model: {exc}")
        return 1
    print("[2/4] LangChain chat model ready")

    try:
        reply = llm.invoke(
            [HumanMessage(content="Reply with exactly the word: ready")]
        )
    except Exception as exc:
        print(f"[3/4] FAILED to reach Gemini: {exc}")
        return 1
    print(f"[3/4] Gemini responded    : {response_text(reply)[:60]!r}")

    sample = ROOT / "samples" / "attention-is-all-you-need.pdf"
    if sample.exists():
        from src.pdf_loader import load_pdf
        from src.summarizer import PDFSummarizer

        document = load_pdf(sample, settings=settings)
        print(
            f"[4/4] Loaded sample     : {document.name} "
            f"({document.page_count} pages, {document.stats.words:,} words)"
        )
        result = PDFSummarizer(settings).summarize(document, style="TL;DR", length="Brief")
        print(f"      summary preview  : {result.summary[:200]}...")
        print(f"      tokens           : {result.total_tokens:,}")
    else:
        print("[4/4] Sample PDF not found - skipping document round-trip")

    print("-" * 62)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
