"""LLM layer: prompt construction, question condensing, and provider backends."""
from __future__ import annotations

import os

from chunker import Chunk
from config import settings

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions about a PDF document.\n"
    "Rules:\n"
    "1. Use ONLY the numbered context passages provided. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I couldn't find that in the document.\"\n"
    "3. Cite the passages you used inline as [1], [2], etc.\n"
    "4. Be concise and accurate; use bullet points for lists."
)

NOT_FOUND = "I couldn't find that in the document."


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, page {c.page})\n{c.text}"
        for i, (c, _) in enumerate(hits, start=1)
    )


def _chat(system: str, messages: list[dict], max_tokens: int) -> str:
    provider = settings.llm_provider

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return (resp.choices[0].message.content or "").strip()

    raise ValueError(f"Unknown LLM provider: {provider}")


def llm_available() -> bool:
    p = settings.llm_provider
    if p == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if p == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return False


def condense_question(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up ("what about its cost?") into a standalone search query."""
    if not history or not llm_available():
        return question
    recent = history[-6:]
    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    prompt = (
        "Given the conversation and a follow-up question, rewrite the follow-up as a "
        "standalone question. Return ONLY the rewritten question.\n\n"
        f"Conversation:\n{transcript}\n\nFollow-up: {question}\n\nStandalone question:"
    )
    try:
        out = _chat("You rewrite questions.", [{"role": "user", "content": prompt}], 120)
        return out or question
    except Exception:  # noqa: BLE001 - retrieval still works with the raw question
        return question


def generate_answer(question: str, hits: list[tuple[Chunk, float]], history: list[dict]) -> str:
    if not llm_available():
        # Extractive fallback so the project still demos without any API key.
        best, _ = hits[0]
        return (
            "(No LLM API key configured - showing the most relevant passage.)\n\n"
            f"{best.text}\n\n[page {best.page}]"
        )

    user_msg = (
        f"Context passages:\n{build_context(hits)}\n\n"
        f"Question: {question}\n\nAnswer using only the context above."
    )
    messages = [*history[-6:], {"role": "user", "content": user_msg}]
    return _chat(SYSTEM_PROMPT, messages, settings.max_tokens)
