"""Token estimates, retrieval metrics, and persistent per-query measurements."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def retrieval_metrics(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], *, k: int
) -> dict[str, float | None]:
    if k < 1:
        raise ValueError("k must be greater than zero")
    relevant = set(relevant_ids)
    if not relevant:
        return {"precision_at_k": None, "recall_at_k": None, "mrr": None}
    ranked = list(retrieved_ids[:k])
    hits = sum(chunk_id in relevant for chunk_id in ranked)
    first_relevant_rank = next(
        (rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant),
        None,
    )
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant),
        "mrr": 1 / first_relevant_rank if first_relevant_rank is not None else 0.0,
    }


def aggregate_retrieval_metrics(
    results: Sequence[dict[str, float | None]],
) -> dict[str, float | None]:
    metric_names = ("precision_at_k", "recall_at_k", "mrr")
    aggregate: dict[str, float | None] = {}
    for name in metric_names:
        values = [float(result[name]) for result in results if result.get(name) is not None]
        aggregate[name] = sum(values) / len(values) if values else None
    return aggregate


def aggregate_groundedness(
    annotations: Sequence[dict[str, int | None]],
) -> float | None:
    total_claims = sum(int(item.get("claims_total") or 0) for item in annotations)
    supported_claims = sum(
        int(item.get("claims_supported") or 0) for item in annotations
    )
    if total_claims == 0:
        return None
    return supported_claims / total_claims


class MetricsStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS query_records "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def record(self, record: dict) -> int:
        timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
        payload = {**record, "timestamp": timestamp}
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO query_records (timestamp, payload) VALUES (?, ?)",
                (timestamp, json.dumps(payload, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 1000) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM query_records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def update(self, record_id: int, record: dict) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE query_records SET payload = ? WHERE id = ?",
                (payload, record_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Query measurement {record_id} does not exist.")

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM query_records")


def average_measurement(records: Sequence[dict], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return sum(values) / len(values) if values else None


def new_query_record() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": "",
        "active_documents": 0,
        "retrieved_chunks": 0,
        "retrieval_latency_ms": 0.0,
        "generation_latency_ms": None,
        "end_to_end_latency_ms": 0.0,
        "embedding_indexing_duration_ms": None,
        "retrieved_chunk_ids": [],
        "distances": [],
        "top_k": None,
        "retrieved_chunk_sizes": [],
        "retrieved_chunk_overlaps": [],
        "context_characters": 0,
        "estimated_context_tokens": 0,
        "estimated_prompt_tokens": None,
        "estimated_answer_tokens": None,
        "estimated_total_tokens": None,
        "actual_provider_usage": None,
        "model_name": None,
        "model_configuration": {"temperature": 0.1, "top_k": None},
        "citation_count": 0,
        "valid_citation_count": 0,
        "invalid_citation_count": 0,
        "citation_reference_validity_rate": None,
        "citation_support_score": None,
        "citation_support_evaluated_claims": 0,
        "supported_claim_count": 0,
        "unsupported_claim_count": 0,
        "precision_at_k": None,
        "recall_at_k": None,
        "mrr": None,
        "groundedness_score": None,
        "answer_status": "not_generated",
        "error": None,
    }


def query_summary(records: Sequence[dict]) -> dict[str, int]:
    query_records = [
        record
        for record in records
        if record.get("answer_status")
        in {"success", "insufficient_context", "api_failure"}
    ]
    return {
        "queries_tested": len(query_records),
        "successful_answers": sum(
            record.get("answer_status") == "success" for record in query_records
        ),
        "insufficient_context_responses": sum(
            record.get("answer_status") == "insufficient_context"
            for record in query_records
        ),
        "api_failures": sum(
            record.get("answer_status") == "api_failure" for record in query_records
        ),
        "benchmark_questions": sum(
            record.get("answer_status") == "retrieval_evaluation" for record in records
        ),
    }
