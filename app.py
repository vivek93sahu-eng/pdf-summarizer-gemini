"""Streamlit front-end for PDF Summarizer AI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import traceback
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    APP_ICON,
    APP_NAME,
    APP_TAGLINE,
    LENGTH_INSTRUCTIONS,
    LANGUAGE_CHOICES,
    MODES,
    MODEL_CHOICES,
    STYLE_INSTRUCTIONS,
    VERSION,
    ConfigError,
    Settings,
)
from src.pdf_loader import PDFDocument, PDFLoadError, load_pdf
from src.qa import DocumentChat
from src.summarizer import PDFSummarizer, SummarizationError, SummaryResult
from src.utils import human_bytes, human_duration, sha256_bytes

st.set_page_config(
    page_title=f"{APP_NAME} · LangChain + Gemini",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, Segoe UI, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1150px; }
#MainMenu, footer { visibility: hidden; }

.hero {
    background: linear-gradient(120deg, #4338ca 0%, #7c3aed 45%, #db2777 100%);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    color: #fff;
    margin-bottom: 1.4rem;
    box-shadow: 0 18px 40px -20px rgba(67, 56, 202, 0.75);
}
.hero h1 { color: #fff; font-size: 2.1rem; font-weight: 800; margin: 0 0 .35rem 0; letter-spacing: -.02em; }
.hero p { color: rgba(255,255,255,.88); font-size: 1.02rem; margin: 0; max-width: 60ch; }
.pill {
    display: inline-block; padding: .22rem .65rem; border-radius: 999px;
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
    font-size: .74rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
    margin-right: .4rem; margin-bottom: .5rem;
}
.doccard {
    border: 1px solid rgba(124,58,237,.25); border-radius: 14px; padding: 1rem 1.1rem;
    background: rgba(124,58,237,.045); margin-bottom: .7rem;
}
.doccard b { font-size: .98rem; }
.doccard span { color: #6b7280; font-size: .84rem; }
.step {
    border: 1px solid rgba(107,114,128,.22); border-radius: 14px; padding: 1.05rem 1.1rem;
    height: 100%; background: rgba(255,255,255,.02);
}
.step h4 { margin: .1rem 0 .35rem 0; font-size: .98rem; }
.step p { color: #6b7280; font-size: .88rem; margin: 0; }
.summary-box {
    border-left: 4px solid #7c3aed; padding: .2rem 0 .2rem 1.1rem; margin-top: .4rem;
}
.q-source { color: #6b7280; font-size: .8rem; }
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
</style>
"""

HERO = f"""
<div class="hero">
  <div><span class="pill">LangChain</span><span class="pill">Gemini</span><span class="pill">Streamlit</span></div>
  <h1>{APP_ICON} {APP_NAME}</h1>
  <p>{APP_TAGLINE} — upload a PDF, get a structured summary, then ask follow-up questions with page-level citations.</p>
</div>
"""


# --------------------------------------------------------------------------- #
# Settings / session helpers
# --------------------------------------------------------------------------- #
def get_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state.settings = Settings.from_env()
    return st.session_state.settings


def load_streamlit_secrets() -> None:
    """Bridge Streamlit Cloud secrets into environment variables.

    Locally the app reads `.env`; on Streamlit Community Cloud secrets are only
    exposed through `st.secrets`, so mirror the keys we care about.
    """
    import os

    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GOOGLE_API_KEY"):
        if os.getenv(key):
            continue
        try:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
        except Exception:  # no secrets.toml present - expected locally
            continue


def store() -> dict:
    return st.session_state.setdefault("results", {})


def doc_cache() -> dict[str, PDFDocument]:
    return st.session_state.setdefault("documents", {})


def chat_cache() -> dict:
    return st.session_state.setdefault("chats", {})


def history_cache() -> dict:
    return st.session_state.setdefault("histories", {})


def page_summary_cache() -> dict:
    return st.session_state.setdefault("page_summaries", {})


def cache_key(name: str, digest: str, opts: dict[str, Any]) -> str:
    signature = json.dumps(opts, sort_keys=True, default=str)
    return f"{digest}:{signature}:{name}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(settings: Settings) -> dict[str, Any]:
    with st.sidebar:
        st.markdown(f"### {APP_ICON} {APP_NAME}")
        st.caption(f"v{VERSION} · LangChain + Google Gemini")

        with st.expander("Model", expanded=True):
            model_label = st.selectbox(
                "Gemini model",
                options=list(MODEL_CHOICES),
                index=0,
                help="Flash models are fast and cheap; Pro is slower but writes better prose.",
            )
            st.caption(f"Configured default: `{settings.model}`")

        st.markdown("**Summary style**")
        style = st.selectbox("Style", list(STYLE_INSTRUCTIONS), label_visibility="collapsed")
        length = st.radio(
            "Length",
            list(LENGTH_INSTRUCTIONS),
            index=1,
            horizontal=True,
            label_visibility="collapsed",
        )
        mode_label = st.selectbox("Strategy", list(MODES), help=(
            "Map-reduce summarises chunks independently and merges them (fast). "
            "Refine carries one running summary through the document (more coherent, slower)."
        ))
        language = st.selectbox("Output language", LANGUAGE_CHOICES)

        with st.expander("Advanced", expanded=False):
            temperature = st.slider(
                "Creativity", 0.0, 1.0, float(settings.temperature), 0.05,
                help="Lower is more literal and consistent.",
            )
            chunk_tokens = st.slider(
                "Chunk size (tokens)", 2000, 14000, int(settings.chunk_tokens), 500,
                help="Smaller chunks are more precise but need more API calls.",
            )
            if st.button("Clear cached results", width="stretch"):
                st.session_state.results = {}
                st.session_state.chats = {}
                st.session_state.histories = {}
                st.session_state.page_summaries = {}
                st.rerun()

        st.divider()
        st.caption(f"API key: `{settings.masked_key}`")
        if st.button("Test Gemini connection", width="stretch"):
            probe_model(settings, MODEL_CHOICES[model_label])
        st.caption("Keys stay local — nothing is uploaded except the extracted text.")

        return {
            "model": MODEL_CHOICES[model_label],
            "style": style,
            "length": length,
            "mode": MODES[mode_label],
            "language": language,
            "temperature": temperature,
            "chunk_tokens": chunk_tokens,
        }


def apply_options(settings: Settings, opts: dict[str, Any]) -> Settings:
    settings.model = opts["model"]
    settings.temperature = opts["temperature"]
    settings.chunk_tokens = opts["chunk_tokens"]
    return settings


def probe_model(settings: Settings, model: str) -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI

    with st.spinner("Calling Gemini..."):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.api_key,
                temperature=0,
                max_output_tokens=32,
                timeout=settings.request_timeout,
            )
            reply = llm.invoke("Reply with exactly: connection ok")
            st.success(f"`{model}` responded: {reply.content}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Connection failed: {exc}")


# --------------------------------------------------------------------------- #
# Upload + summarisation
# --------------------------------------------------------------------------- #
class ProgressReporter:
    """Bridges summarizer callbacks to Streamlit widgets."""

    def __init__(self, status, bar, caption):
        self.status = status
        self.bar = bar
        self.caption = caption

    def __call__(self, stage: str, current: int, total: int, message: str) -> None:
        fraction = 0.0 if not total else min(1.0, max(0.0, current / total))
        self.bar.progress(fraction)
        self.caption.markdown(f"**{stage}** · {message}")
        if stage == "Done":
            self.status.update(label="Summary ready", state="complete", expanded=False)


def render_uploader(settings: Settings) -> list[PDFDocument]:
    uploaded = st.file_uploader(
        "Drop one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Up to {settings.max_file_mb} MB and {settings.max_pages} pages per file.",
    )
    if not uploaded:
        return []

    documents: list[PDFDocument] = []
    cache = doc_cache()
    for file in uploaded:
        data = file.getvalue()
        digest = sha256_bytes(data)
        if digest in cache:
            documents.append(cache[digest])
            continue
        try:
            document = load_pdf(data, file.name, settings)
        except PDFLoadError as exc:
            st.error(str(exc))
            continue
        cache[digest] = document
        documents.append(document)
    return documents


def render_document_cards(documents: list[PDFDocument]) -> None:
    for document in documents:
        stats = document.stats
        extras = []
        if document.looks_scanned:
            extras.append("⚠️ mostly images — OCR recommended")
        if document.empty_pages:
            extras.append(f"{len(document.empty_pages)} page(s) without text")
        note = f"<br><span>{' · '.join(extras)}</span>" if extras else ""
        st.markdown(
            f"""
            <div class="doccard">
              <b>{document.name}</b><br>
              <span>{stats.pages} pages · {stats.words:,} words · {human_bytes(document.size_bytes)}</span>
              {note}
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_summaries(
    documents: list[PDFDocument], settings: Settings, opts: dict[str, Any]
) -> None:
    results = store()
    for position, document in enumerate(documents, start=1):
        key = cache_key(document.name, sha256_bytes(document.data), opts)
        if key in results:
            continue

        label = f"Summarising {document.name} ({position}/{len(documents)})"
        with st.status(label, expanded=True) as status:
            bar = st.progress(0.0)
            caption = st.empty()
            reporter = ProgressReporter(status, bar, caption)
            summarizer = PDFSummarizer(settings, on_progress=reporter)
            try:
                result = summarizer.summarize(
                    document,
                    style=opts["style"],
                    length=opts["length"],
                    mode=opts["mode"],
                    language=opts["language"],
                )
                results[key] = {"document": document, "result": result}
                status.update(label=f"✅ {document.name} summarised", state="complete")
            except SummarizationError as exc:
                status.update(label=f"❌ {document.name} failed", state="error")
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                status.update(label=f"❌ {document.name} failed", state="error")
                st.error(f"Unexpected error: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def metrics_row(result: SummaryResult, document: PDFDocument) -> None:
    stats = document.stats
    cols = st.columns(5)
    cols[0].metric("Pages", stats.pages)
    cols[1].metric("Words", f"{stats.words:,}")
    cols[2].metric("Chunks", result.chunk_count)
    cols[3].metric("Time", human_duration(result.elapsed_seconds))
    cols[4].metric("Tokens", f"{result.total_tokens:,}")


def build_markdown_report(document: PDFDocument, result: SummaryResult) -> str:
    stats = document.stats
    return f"""# Summary: {document.title}

> Generated with **{APP_NAME}** using `{result.model}` on {result.created_at}

| | |
|---|---|
| Source | `{document.name}` |
| Pages | {stats.pages} |
| Words | {stats.words:,} |
| Style | {result.style} |
| Length | {result.length} |
| Strategy | {result.mode} |
| Chunks | {result.chunk_count} |
| Tokens (in/out) | {result.input_tokens:,} / {result.output_tokens:,} |

## Summary

{result.summary}

---
_Generated by {APP_NAME} v{VERSION} · LangChain + Google Gemini_
"""


def render_summary_tab(document: PDFDocument, result: SummaryResult) -> None:
    metrics_row(result, document)
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown(result.summary)
    st.markdown("</div>", unsafe_allow_html=True)

    stem = document.name.rsplit(".", 1)[0]
    report = build_markdown_report(document, result)
    payload = json.dumps(
        {
            "document": document.name,
            "model": result.model,
            "style": result.style,
            "length": result.length,
            "mode": result.mode,
            "language": result.language,
            "summary": result.summary,
            "chunk_count": result.chunk_count,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "created_at": result.created_at,
        },
        indent=2,
    )

    col_dl1, col_dl2, _ = st.columns([1, 1, 3])
    col_dl1.download_button(
        "⬇️ Markdown",
        data=report.encode("utf-8"),
        file_name=f"{stem}.summary.md",
        mime="text/markdown",
        width="stretch",
    )
    col_dl2.download_button(
        "⬇️ JSON",
        data=payload.encode("utf-8"),
        file_name=f"{stem}.summary.json",
        mime="application/json",
        width="stretch",
    )

    if result.partial_summaries:
        with st.expander(f"Intermediate summaries ({len(result.partial_summaries)})"):
            for index, partial in enumerate(result.partial_summaries, start=1):
                st.markdown(f"**Step {index}**")
                st.markdown(partial)
                st.divider()


def render_chat_tab(document: PDFDocument, settings: Settings, key: str) -> None:
    chats = chat_cache()
    histories = history_cache()

    if key not in chats:
        chats[key] = DocumentChat(document, settings)
    chat = chats[key]
    history = histories.setdefault(key, [])

    st.caption("Answers are grounded in this document only — each reply cites the pages it used.")

    if not history:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for index, question in enumerate(chat.suggested_questions()):
            if cols[index % 2].button(question, key=f"suggest-{key}-{index}", width="stretch"):
                st.session_state["pending_question"] = question
                st.rerun()

    for turn in history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])
            if turn.get("sources"):
                pages = ", ".join(f"p. {p}" for p in turn["sources"])
                st.markdown(f'<span class="q-source">Sources: {pages}</span>', unsafe_allow_html=True)

    with st.form(key=f"qa-form-{key}", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about this document",
            value=st.session_state.pop("pending_question", ""),
            placeholder="e.g. What methodology did the authors use?",
        )
        submitted = st.form_submit_button("Ask", width="stretch")

    if submitted and question.strip():
        history.append({"role": "user", "content": question})
        with st.spinner("Searching the document..."):
            try:
                answer, sources = chat.answer(
                    question, [(turn["role"], turn["content"]) for turn in history[:-1]]
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not answer: {exc}")
                return
        history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()


def render_pages_tab(document: PDFDocument, settings: Settings, key: str) -> None:
    pages = [p.number for p in document.pages]
    page_number = st.selectbox("Page", pages, key=f"page-pick-{key}")
    page = next(p for p in document.pages if p.number == page_number)

    if not page.text:
        st.info("This page has no extractable text (it may be a scanned image).")
    else:
        st.text_area("Extracted text", page.text, height=280, key=f"page-text-{key}-{page_number}")

    if st.button("Summarise this page", key=f"page-sum-{key}"):
        cache = page_summary_cache()
        cache_key_ = f"{key}:{page_number}"
        if cache_key_ not in cache:
            with st.spinner("Summarising page..."):
                try:
                    cache[cache_key_] = PDFSummarizer(settings).summarize_page(document, page_number)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not summarise page: {exc}")
        if cache_key_ in cache:
            st.markdown("**Page summary**")
            st.markdown(cache[cache_key_])


def render_raw_tab(document: PDFDocument) -> None:
    st.caption(
        f"{document.stats.words:,} words · {document.stats.characters:,} characters · "
        f"{len(document.empty_pages)} page(s) without text"
    )
    st.text_area("Full extracted text", document.plain_text, height=420, label_visibility="collapsed")
    st.download_button(
        "⬇️ Download extracted text",
        data=document.plain_text.encode("utf-8"),
        file_name=f"{document.name.rsplit('.', 1)[0]}.txt",
        mime="text/plain",
    )


def render_results(documents: list[PDFDocument], settings: Settings, opts: dict[str, Any]) -> None:
    results = store()
    available = []
    for document in documents:
        key = cache_key(document.name, sha256_bytes(document.data), opts)
        if key in results:
            available.append((key, document, results[key]["result"]))

    if not available:
        return

    if len(available) > 1:
        labels = [doc.name for _, doc, _ in available]
        choice = st.selectbox("Showing result for", labels, key="result-picker")
        selected = available[labels.index(choice)]
    else:
        selected = available[0]

    key, document, result = selected
    st.markdown(f"### {document.name}")
    st.caption(
        f"{result.style} · {result.length} · {result.mode.replace('_', '-')} · "
        f"`{result.model}` · {result.created_at}"
    )

    tab_summary, tab_chat, tab_pages, tab_text = st.tabs(
        ["📝 Summary", "💬 Ask the document", "📑 Pages", "🔎 Extracted text"]
    )
    with tab_summary:
        render_summary_tab(document, result)
    with tab_chat:
        render_chat_tab(document, settings, key)
    with tab_pages:
        render_pages_tab(document, settings, key)
    with tab_text:
        render_raw_tab(document)


# --------------------------------------------------------------------------- #
# Empty state
# --------------------------------------------------------------------------- #
def render_empty_state() -> None:
    st.markdown("#### How it works")
    steps = [
        ("1 · Upload", "Drop a PDF — research paper, report, contract, lecture notes."),
        ("2 · Summarise", "LangChain splits the text and Gemini maps, then reduces it into one summary."),
        ("3 · Ask", "Follow up in plain English. Every answer cites the pages it used."),
    ]
    for column, (title, body) in zip(st.columns(3), steps):
        column.markdown(
            f'<div class="step"><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info(
        "🔒 Your PDF never leaves your machine: only the extracted text is sent to the "
        "Gemini API. The key lives in `.env` and is never committed."
    )

    with st.expander("Supported inputs & limits"):
        st.markdown(
            """
            - Text-based PDFs (scanned documents need OCR first)
            - Up to 50 MB and 500 pages per file, several files at once
            - English and 7 other output languages
            - Both **map-reduce** (fast) and **refine** (more coherent) strategies
            - Exports as Markdown or JSON
            """
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)

    load_streamlit_secrets()
    settings = get_settings()

    if not settings.api_key:
        st.error(
            "**GEMINI_API_KEY is not set.** Create a `.env` file next to `app.py` with "
            "`GEMINI_API_KEY=your_key` and restart the app."
        )
        st.code("GEMINI_API_KEY=your_google_ai_studio_api_key", language="bash")
        st.stop()

    try:
        settings.validate()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()

    opts = render_sidebar(settings)
    apply_options(settings, opts)

    documents = render_uploader(settings)
    if not documents:
        render_empty_state()
        return

    render_document_cards(documents)

    with st.container():
        cols = st.columns([1, 1, 2])
        summarise = cols[0].button("✨ Summarise", type="primary", width="stretch")
        clear = cols[1].button("Reset results", width="stretch")
        cols[2].caption(
            f"{len(documents)} document(s) ready · {opts['style']} · {opts['length']}"
        )

    if clear:
        st.session_state.results = {}
        st.session_state.chats = {}
        st.session_state.histories = {}
        st.session_state.page_summaries = {}
        st.rerun()

    if summarise:
        run_summaries(documents, settings, opts)
        st.rerun()

    render_results(documents, settings, opts)


if __name__ == "__main__":
    main()
