"""Prompt templates used by the summarisation and Q&A chains."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are an expert document analyst. You read carefully, never invent facts, "
    "and write in clear, plain language. When the source material is ambiguous, "
    "say so instead of guessing."
)

MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "You are summarising part {index} of {total} of a longer document.\n"
            "Extract only what this part actually says. Keep names, numbers, dates "
            "and technical terms intact. Do not add commentary about the document "
            "as a whole.\n\n"
            "--- PART {index} START (page {page}) ---\n"
            "{text}\n"
            "--- PART {index} END ---\n\n"
            "Write a dense factual summary of this part in 120-200 words.",
        ),
    ]
)

REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Below are summaries of consecutive parts of the document "
            "\"{title}\". Combine them into one coherent summary.\n\n"
            "Style: {style_instruction}\n"
            "Length: {length_instruction}\n"
            "Language: {language_instruction}\n\n"
            "Remove duplication, keep every important fact, and do not mention "
            "that you were given partial summaries.\n\n"
            "--- PART SUMMARIES START ---\n"
            "{summaries}\n"
            "--- PART SUMMARIES END ---",
        ),
    ]
)

DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Summarise the following document, titled \"{title}\".\n\n"
            "Style: {style_instruction}\n"
            "Length: {length_instruction}\n"
            "Language: {language_instruction}\n\n"
            "Base every statement on the document. If it contains no meaningful "
            "content, say so plainly.\n\n"
            "--- DOCUMENT START ---\n"
            "{text}\n"
            "--- DOCUMENT END ---",
        ),
    ]
)

REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "You are refining a running summary of the document \"{title}\".\n\n"
            "Current summary:\n"
            "{summary}\n\n"
            "Next part of the document (page {page}):\n"
            "--- PART START ---\n"
            "{text}\n"
            "--- PART END ---\n\n"
            "Update the summary so it also reflects this new part. Keep the same "
            "style and length:\n"
            "Style: {style_instruction}\n"
            "Length: {length_instruction}\n"
            "Language: {language_instruction}\n\n"
            "Return only the updated summary.",
        ),
    ]
)

PAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Summarise page {page} of the document \"{title}\" in 2-4 sentences. "
            "If the page has no usable text, reply exactly: (no extractable text)",
        ),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a PDF using only the provided excerpts. "
            "Cite the page numbers you used, like (p. 4). If the excerpts do not "
            "contain the answer, say you could not find it in the document - never "
            "guess. Be concise unless the question asks for detail.",
        ),
        (
            "human",
            "Document: {title}\n\n"
            "Excerpts:\n{context}\n\n"
            "Conversation so far:\n{history}\n\n"
            "Question: {question}",
        ),
    ]
)



def style_instruction(style: str) -> str:
    from .config import STYLE_INSTRUCTIONS

    return STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["Executive brief"])


def length_instruction(length: str) -> str:
    from .config import LENGTH_INSTRUCTIONS

    return LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["Standard"])


def language_instruction(language: str) -> str:
    if not language or language == "Same as document":
        return "Write in the same language as the document."
    return f"Write the answer in {language}, even if the document is in another language."
