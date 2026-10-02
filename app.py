"""Streamlit chat interface for the document Q&A system.

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag import config
from rag.ingest import delete_source, ingest_files, list_sources
from rag.query import ask

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------- #
# Sidebar: documents
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded and st.button("Index uploaded files", type="primary"):
        with st.spinner("Extracting, chunking and embedding..."):
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for f in uploaded:
                    p = Path(tmp) / f.name
                    p.write_bytes(f.getbuffer())
                    paths.append(p)
                rep = ingest_files(paths)
        st.success(f"Indexed {rep['files']} file(s), {rep['chunks']} chunks")
        for s in rep["skipped"]:
            st.warning(s)

    sources = list_sources()
    st.caption(f"{len(sources)} document(s) indexed")
    selected = st.multiselect("Restrict questions to", sources, default=[], help="Empty = search all documents")
    for s in sources:
        c1, c2 = st.columns([4, 1])
        c1.write(f"📄 {s}")
        if c2.button("✕", key=f"del_{s}", help=f"Remove {s} from the index"):
            delete_source(s)
            st.rerun()

    st.divider()
    top_k = st.slider("Passages per question", 2, 10, config.TOP_K)
    mode = f"OpenAI ({config.CHAT_MODEL}, embeddings {config.EMBEDDING_MODEL})" if config.llm_available() else "LOCAL (MiniLM embeddings, extractive answers — set OPENAI_API_KEY for generated answers)"
    st.caption(f"Mode: {mode}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
st.title("📄 Document Q&A")
st.caption("Upload documents on the left, then ask questions. Answers cite the passages they come from.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("passages"):
            with st.expander("Sources"):
                for p in m["passages"]:
                    loc = f"{p['source']}, page {p['page']}" if p["page"] else p["source"]
                    st.markdown(f"**[{p['rank']}] {loc}** · similarity {p['score']}")
                    st.text(p['text'][:600] + ('…' if len(p['text']) > 600 else ''))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            ans = ask(question, top_k=top_k, sources=selected or None, history=history)
        st.markdown(ans.text)
        passages = [p.__dict__ for p in ans.passages]
        if passages:
            with st.expander("Sources"):
                for p in passages:
                    loc = f"{p['source']}, page {p['page']}" if p["page"] else p["source"]
                    st.markdown(f"**[{p['rank']}] {loc}** · similarity {p['score']}")
                    st.text(p['text'][:600] + ('…' if len(p['text']) > 600 else ''))
    st.session_state.messages.append({"role": "assistant", "content": ans.text, "passages": passages})
