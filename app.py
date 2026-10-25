"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from rag.config import settings
from rag.embeddings import Embedder
from rag.llm import llm_available
from rag.pipeline import PDFQAPipeline

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> Embedder:
    return Embedder(settings.embedding_model)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = PDFQAPipeline(get_embedder())
    st.session_state.messages = []

pipeline: PDFQAPipeline = st.session_state.pipeline

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("📄 Documents")
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs", type="primary", disabled=not files):
        with st.spinner("Reading, chunking and embedding..."):
            for f in files:
                if f.name in pipeline.documents:
                    continue
                try:
                    n = pipeline.add_pdf(f, f.name)
                    st.success(f"{f.name}: {n} chunks indexed")
                except ValueError as e:
                    st.error(str(e))

    if pipeline.documents:
        st.caption("Indexed: " + ", ".join(pipeline.documents))
        if st.button("Clear all"):
            pipeline.reset()
            st.session_state.messages = []
            st.rerun()

    st.divider()
    if llm_available():
        st.caption(f"LLM: {settings.llm_provider}")
    else:
        st.warning("No API key found. Answers will show the best matching passage only. "
                   "Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in .env for generated answers.")

# ---------------- chat ----------------
st.title("GenAI PDF Question Answering Assistant")
st.caption("Retrieval-Augmented Generation: answers are grounded in your document, with page citations.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about your PDF...", disabled=not pipeline.ready):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            answer = pipeline.ask(question, history)
        st.markdown(answer.text)
        if answer.sources:
            with st.expander(f"Sources ({len(answer.sources)})"):
                for i, (chunk, score) in enumerate(answer.sources, start=1):
                    st.markdown(f"**[{i}] {chunk.source} - page {chunk.page}** (similarity {score:.2f})")
                    st.caption(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
    st.session_state.messages.append({"role": "assistant", "content": answer.text})

if not pipeline.ready:
    st.info("Upload a PDF in the sidebar and click **Process PDFs** to begin.")
