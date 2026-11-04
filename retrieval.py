"""Semantic retrieval and stable per-answer source identifiers."""

from __future__ import annotations

import time
from collections.abc import Sequence

from schemas import RetrievedChunk


def retrieve(
    question: str,
    *,
    store: object,
    embedder: object,
    top_k: int = 4,
) -> tuple[list[RetrievedChunk], float]:
    if not question.strip():
        raise ValueError("Enter a question before retrieving documents.")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    started = time.perf_counter()
    query_vector = embedder.embed_query(question)
    rows = store.search(query_vector, top_k)
    rows.sort(key=lambda row: row["distance"])
    results = [
        RetrievedChunk(
            document_id=row["document_id"],
            filename=row["filename"],
            page_number=None if int(row["page_number"]) < 1 else int(row["page_number"]),
            chunk_id=row["chunk_id"],
            content_hash=row["content_hash"],
            text=row["text"],
            distance=float(row["distance"]),
            citation_id=f"S{index}",
            chunk_size=int(row.get("chunk_size", 800)),
            chunk_overlap=int(row.get("chunk_overlap", 120)),
        )
        for index, row in enumerate(rows, start=1)
    ]
    return results, (time.perf_counter() - started) * 1000


def has_sufficient_relevance(
    results: Sequence[RetrievedChunk], *, max_distance: float = 0.8
) -> bool:
    return bool(results) and results[0].distance <= max_distance
