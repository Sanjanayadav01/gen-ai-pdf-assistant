"""Text chunking with overlap. Prefers to cut at paragraph / line / sentence / word boundaries."""
import re
from dataclasses import dataclass, asdict

from .loader import PageText


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


_SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks: list[str] = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:  # don't make tiny chunks
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int, overlap: int, start_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = start_id
    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(next_id, page.source, page.page, piece))
            next_id += 1
    return chunks
