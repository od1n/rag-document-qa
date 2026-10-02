"""Question answering: retrieve the most relevant passages, then answer with citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .store import get_collection

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the passages provided. "
    "If the passages do not contain the answer, say so plainly instead of guessing. "
    "Cite the passages you used as [1], [2]... matching their numbers. "
    "Answer in the same language as the question. Be concise."
)


@dataclass
class Passage:
    rank: int
    text: str
    source: str
    page: int
    score: float  # cosine similarity (1 = identical)


@dataclass
class Answer:
    text: str
    passages: list[Passage]
    mode: str  # "llm" or "extractive"


def retrieve(question: str, top_k: int = config.TOP_K, sources: list[str] | None = None) -> list[Passage]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    where = {"source": {"$in": sources}} if sources else None
    res = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
        passages.append(Passage(rank=i, text=doc, source=meta["source"], page=int(meta.get("page", 0)), score=round(1 - dist, 3)))
    return passages


def _context_block(passages: list[Passage]) -> str:
    parts = []
    for p in passages:
        loc = f"{p.source}, page {p.page}" if p.page else p.source
        parts.append(f"[{p.rank}] ({loc})\n{p.text}")
    return "\n\n".join(parts)


def answer_with_llm(question: str, passages: list[Passage], history: list[dict] | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # short conversational memory (last turns), without the old passages
    for turn in (history or [])[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"PASSAGES:\n\n{_context_block(passages)}\n\nQUESTION: {question}",
    })
    resp = client.chat.completions.create(model=config.CHAT_MODEL, messages=messages, temperature=0.1)
    return resp.choices[0].message.content.strip()


def answer_extractive(question: str, passages: list[Passage]) -> str:
    """No LLM available: return the best passages verbatim so the user still gets the facts."""
    if not passages:
        return "No indexed documents match this question."
    lines = ["No OpenAI key configured, so here are the most relevant passages instead of a generated answer:\n"]
    for p in passages[:3]:
        loc = f"{p.source}, page {p.page}" if p.page else p.source
        flat = re.sub(r"(^|\s)(\d+)\.\s", r"\1\2\\. ", p.text.replace("\n", " "))  # keep "1. x" from becoming a list
        quoted = "> " + flat
        lines.append(f"**[{p.rank}]** {loc} · similarity {p.score}\n{quoted}\n")
    return "\n".join(lines)


def ask(question: str, top_k: int = config.TOP_K, sources: list[str] | None = None, history: list[dict] | None = None) -> Answer:
    passages = retrieve(question, top_k=top_k, sources=sources)
    if not passages:
        return Answer(text="No documents are indexed yet. Upload a PDF first.", passages=[], mode="none")
    if config.llm_available():
        return Answer(text=answer_with_llm(question, passages, history), passages=passages, mode="llm")
    return Answer(text=answer_extractive(question, passages), passages=passages, mode="extractive")
