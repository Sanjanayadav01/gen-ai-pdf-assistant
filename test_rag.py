import numpy as np
import pytest

from rag.chunker import Chunk, chunk_pages, split_text
from rag.loader import PageText


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", 1000, 200) == ["Hello world."]


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_text(text, 300, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word1999")


def test_overlap_shares_text_between_neighbours():
    text = " ".join(f"w{i}" for i in range(600))
    chunks = split_text(text, 200, 60)
    tail = chunks[0].split()[-3:]
    assert all(t in chunks[1] for t in tail)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)


def test_chunk_pages_keep_page_numbers_and_ids():
    pages = [PageText("a.pdf", 1, "x " * 400), PageText("a.pdf", 2, "y " * 400)]
    chunks = chunk_pages(pages, 300, 50)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_vectorstore_search_and_persistence(tmp_path):
    pytest.importorskip("faiss")
    from rag.vectorstore import VectorStore

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [Chunk(i, "d.pdf", i + 1, f"text {i}") for i in range(5)]

    store = VectorStore(8)
    store.add(chunks, vecs)
    top = store.search(vecs[3], k=2)
    assert top[0][0].chunk_id == 3 and top[0][1] == pytest.approx(1.0, abs=1e-4)

    store.save(str(tmp_path))
    reloaded = VectorStore.load(str(tmp_path))
    assert len(reloaded) == 5
    assert reloaded.search(vecs[1], k=1)[0][0].page == 2
