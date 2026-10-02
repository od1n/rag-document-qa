"""Minimal, production-sane RAG for document Q&A.

Pipeline: PDF/TXT/MD -> text -> overlapping chunks -> embeddings -> ChromaDB
          question -> embedding -> top-k chunks -> LLM answer with citations
"""
