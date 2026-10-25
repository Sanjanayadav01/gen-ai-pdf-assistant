"""FAISS vector store with JSON metadata and on-disk persistence."""
import json
import os

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # exact search; fine for thousands of chunks
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            return
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        k = min(k, len(self.chunks))
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(q, k)
        return [
            (self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1
        ]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), encoding="utf-8") as f:
            chunks = [Chunk(**d) for d in json.load(f)]
        store = cls(index.d)
        store.index = index
        store.chunks = chunks
        return store
