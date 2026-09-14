"""
semantic_search.py

V1 — Baseline retrieval.

    Query -> Embedding -> Vector search -> Top-k memories

Pure cosine-similarity search over stored memory embeddings. No notion of
recency, importance, or scope/context — every memory is equally "eligible"
regardless of when it was written, how important it was, or which
user/session/project it belongs to.

This is intentionally simple so its failure modes are visible: e.g. an old,
low-importance memory from an unrelated project can outrank a highly
relevant, recent, important one purely because it shares more surface-level
vocabulary with the query. contextual_retrieval.py (V2) fixes this.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingModel, get_default_embedding_model


@dataclass
class Memory:
    """A single unit of stored knowledge about/for a user."""

    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    # Metadata used by V2 scoring; V1 ignores all of it.
    scope: str = "general"            # e.g. "user/123/project/rag-app"
    importance: float = 0.5           # 0.0 - 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """In-memory vector store. Swap for FAISS/pgvector/etc. later —
    the interface (add_memory / search) should stay stable."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or get_default_embedding_model()
        self._memories: List[Memory] = []

    def add_memory(self, memory: Memory) -> Memory:
        if memory.embedding is None:
            memory.embedding = self.embedding_model.embed(memory.text)
        self._memories.append(memory)
        return memory

    def add_memories(self, memories: List[Memory]) -> List[Memory]:
        return [self.add_memory(m) for m in memories]

    @property
    def memories(self) -> List[Memory]:
        return list(self._memories)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Pure semantic search — V1 baseline."""
        query_embedding = self.embedding_model.embed(query)
        scored = [
            (mem, cosine_similarity(query_embedding, mem.embedding))
            for mem in self._memories
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
    