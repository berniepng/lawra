#!/usr/bin/env python3
"""
Lawra Working Memory — ChromaDB-backed semantic conversation store.

Stores Q&A pairs per session and retrieves the most semantically
relevant past exchanges to inject as context for the current query.

Why ChromaDB over plain list:
  - Semantic retrieval: finds related past exchanges, not just recent ones
  - Persistent across server restarts
  - Session-isolated: users don't see each other's history
  - Scalable: handles hundreds of exchanges without prompt overflow

ChromaDB runs in-process (no extra Docker service needed).
Data persists in .chromadb/ in the frontend directory.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PERSIST_DIR  = Path(__file__).parent / ".chromadb"
COLLECTION   = "lawra_conversations"
OLLAMA_URL   = "http://localhost:11434"
EMBED_MODEL  = "nomic-embed-text"
MAX_MEMORY   = 3      # max past exchanges to inject per query
SNIPPET_LEN  = 300    # chars of answer to store (keeps ChromaDB lean)


def _make_client():
    import chromadb
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


def _make_ef():
    """Ollama embedding function — same model as the main pipeline."""
    try:
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        return OllamaEmbeddingFunction(
            url=f"{OLLAMA_URL}/api/embeddings",
            model_name=EMBED_MODEL,
        )
    except (ImportError, Exception):
        # Fallback: let ChromaDB use its default (requires sentence-transformers)
        return None


class LawraMemory:
    """Semantic session memory backed by ChromaDB."""

    def __init__(self):
        self._client     = _make_client()
        self._ef         = _make_ef()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────
    def store(self, session_id: str, query: str, answer: str,
              sources: Optional[list] = None) -> None:
        """Store a Q&A exchange for a session."""
        doc_id  = f"{session_id}_{uuid.uuid4().hex[:8]}"
        # The document text combines Q + A for semantic search
        doc     = f"Question: {query}\nAnswer: {answer[:SNIPPET_LEN]}"
        ts      = datetime.now(timezone.utc).isoformat()

        source_labels = ", ".join(
            f"{s.get('title','')} {s.get('section_label','')}"
            for s in (sources or [])[:3]
        )

        self._collection.add(
            documents=[doc],
            ids=[doc_id],
            metadatas=[{
                "session_id":    session_id,
                "query":         query[:200],
                "answer":        answer[:SNIPPET_LEN],
                "source_labels": source_labels,
                "timestamp":     ts,
            }],
        )

    # ── Read ──────────────────────────────────────────────────────────────
    def search(self, session_id: str, query: str,
               n: int = MAX_MEMORY) -> list[dict]:
        """
        Return the N most semantically relevant past exchanges
        for this session. Returns [] if no history yet.
        """
        try:
            count = self._collection.count()
        except Exception:
            return []

        if count == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n, count),
                where={"session_id": session_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        exchanges = []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        dists     = results.get("distances",  [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = max(0.0, 1.0 - dist)
            if similarity < 0.3:        # skip low-relevance exchanges
                continue
            exchanges.append({
                "query":         meta.get("query", ""),
                "answer":        meta.get("answer", ""),
                "source_labels": meta.get("source_labels", ""),
                "timestamp":     meta.get("timestamp", ""),
                "similarity":    round(similarity, 3),
            })

        return exchanges

    def format_context(self, exchanges: list[dict]) -> str:
        """Format retrieved memory into a concise context block for the LLM."""
        if not exchanges:
            return ""
        lines = ["Relevant earlier exchanges in this conversation:"]
        for ex in exchanges:
            lines.append(f"  Q: {ex['query']}")
            lines.append(f"  A: {ex['answer']}")
            if ex.get("source_labels"):
                lines.append(f"  (sources: {ex['source_labels']})")
            lines.append("")
        return "\n".join(lines).strip()

    def session_count(self, session_id: str) -> int:
        """Number of stored exchanges for a session."""
        try:
            results = self._collection.get(where={"session_id": session_id})
            return len(results.get("ids", []))
        except Exception:
            return 0

    def clear_session(self, session_id: str) -> int:
        """Delete all memory for a session. Returns number deleted."""
        try:
            results = self._collection.get(where={"session_id": session_id})
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0


# ── Module-level singleton (shared across server.py requests) ─────────────────
_memory_instance: Optional[LawraMemory] = None

def get_memory() -> LawraMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LawraMemory()
    return _memory_instance
