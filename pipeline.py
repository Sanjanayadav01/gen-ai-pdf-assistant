"""End-to-end RAG pipeline: ingest PDFs -> retrieve -> generate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import BinaryIO, Union

from .chunker import Chunk, chunk_pages
from .config import settings
from .embeddings import Embedder
from .llm import NOT_FOUND, condense_question, generate_answer
from .loader import load_pdf
from .vectorstore import VectorStore


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]] = field(default_factory=list)
    search_query: str = ""


class PDFQAPipeline:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.store: VectorStore | None = None
        self.documents: list[str] = []

    # ---- ingestion ---------------------------------------------------------
    def add_pdf(self, file: Union[str, BinaryIO], name: str) -> int:
        """Index a PDF. Returns the number of chunks added."""
        pages = load_pdf(file, name)
        start_id = len(self.store) if self.store else 0
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap, start_id)
        vectors = self.embedder.embed([c.text for c in chunks])
        if self.store is None:
            self.store = VectorStore(vectors.shape[1])
        self.store.add(chunks, vectors)
        if name not in self.documents:
            self.documents.append(name)
        return len(chunks)

    @property
    def ready(self) -> bool:
        return self.store is not None and len(self.store) > 0

    def reset(self) -> None:
        self.store = None
        self.documents = []

    # ---- question answering ------------------------------------------------
    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        if not self.ready:
            return Answer("Please upload and process a PDF first.")
        history = history or []

        query = condense_question(question, history)
        qvec = self.embedder.embed([query])[0]
        hits = [h for h in self.store.search(qvec, settings.top_k) if h[1] >= settings.min_score]

        if not hits:
            return Answer(NOT_FOUND, [], query)
        return Answer(generate_answer(question, hits, history), hits, query)

    # ---- persistence -------------------------------------------------------
    def save(self, directory: str | None = None) -> None:
        if self.ready:
            self.store.save(directory or settings.index_dir)

    def load(self, directory: str | None = None) -> None:
        self.store = VectorStore.load(directory or settings.index_dir)
        self.documents = sorted({c.source for c in self.store.chunks})
