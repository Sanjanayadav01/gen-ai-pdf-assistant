# GenAI-Powered PDF Question Answering Assistant (RAG)

Upload one or more PDFs and ask questions in natural language. The assistant retrieves the most relevant passages from your document and uses a Large Language Model (LLM) to write an answer **grounded in that text**, with page-level citations. If the answer is not in the document, it says so instead of guessing.

## 1. Problem and motivation
LLMs do not know the contents of your private documents and can hallucinate. **Retrieval-Augmented Generation (RAG)** fixes this by fetching relevant text at question time and giving it to the model as context.

## 2. Architecture

```
            INGESTION (once per PDF)
PDF -> pypdf (text per page) -> chunker (1000 chars, 200 overlap)
    -> sentence-transformers embeddings -> FAISS index (+ chunk metadata)

            QUERY (every question)
Question -> [condense follow-up using chat history]
         -> embed -> FAISS top-k similarity search -> score threshold
         -> prompt = system rules + numbered context + question
         -> LLM (Claude / OpenAI) -> answer with [1][2] citations -> UI shows sources
```

| Module | Responsibility |
|---|---|
| `rag/loader.py` | Extract text page by page (keeps page numbers) |
| `rag/chunker.py` | Overlapping chunks that break on paragraph/sentence boundaries |
| `rag/embeddings.py` | Local `all-MiniLM-L6-v2` embeddings (384-dim, normalised) |
| `rag/vectorstore.py` | FAISS `IndexFlatIP` (cosine similarity), save/load to disk |
| `rag/llm.py` | Prompting, follow-up question rewriting, provider backends |
| `rag/pipeline.py` | Orchestrates ingestion, retrieval and generation |
| `app.py` | Streamlit chat UI with source viewer |
| `cli.py` | Terminal interface |
| `tests/` | Unit tests for chunking and vector store |

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add your API key
```

## 4. Run

```bash
streamlit run app.py            # web UI
python cli.py sample.pdf        # terminal
pytest -q                       # tests
```

Without an API key the app still runs in demo mode and shows the best matching passage; with a key it generates full answers.

## 5. Key design decisions (good viva answers)
- **Chunk size 1000 / overlap 200**: large enough to hold a complete idea, overlap prevents answers being split across a boundary.
- **Normalised embeddings + inner product** = cosine similarity.
- **Similarity threshold (`MIN_SCORE`)**: if nothing relevant is retrieved, the LLM is never called, which reduces hallucination and cost.
- **Strict system prompt**: "use only the context", cite sources, admit when unknown.
- **Query condensing**: "What about its cost?" is rewritten into a standalone question before retrieval, so multi-turn chat works.
- **Page metadata** travels with each chunk, enabling verifiable citations.
- **Provider-agnostic LLM layer**: switch between Claude and OpenAI with one env variable.

## 6. Limitations and future work
- Scanned PDFs need OCR (e.g. `pytesseract`); currently reported with a clear error.
- Tables and multi-column layouts can extract imperfectly.
- Exact search is fine up to ~100k chunks; beyond that use an IVF/HNSW index.
- Improvements: hybrid BM25 + vector search, cross-encoder re-ranking, streaming responses, evaluation with a question/answer set (RAGAS-style faithfulness and relevance metrics), Docker deployment.

## 7. Suggested internship report outline
1. Introduction and objectives  2. Literature/background (LLMs, embeddings, RAG)  3. System design (diagram above)  4. Implementation  5. Testing and sample Q&A screenshots  6. Results and limitations  7. Future scope.
