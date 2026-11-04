"""Evaluation of retrieval against human-labeled chunk identifiers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from metrics import aggregate_groundedness, aggregate_retrieval_metrics, retrieval_metrics


def load_dataset(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("The evaluation dataset must be a JSON list.")
    for index, row in enumerate(data):
        if not isinstance(row, dict) or not row.get("question"):
            raise ValueError(f"Evaluation row {index + 1} must include a question.")
        relevant_ids = row.get("relevant_chunk_ids", [])
        if not isinstance(relevant_ids, (list, str)):
            raise ValueError(f"Evaluation row {index + 1} has invalid relevant chunk IDs.")
        claims_total = int(row.get("claims_total") or 0)
        claims_supported = int(row.get("claims_supported") or 0)
        if claims_total < 0 or claims_supported < 0 or claims_supported > claims_total:
            raise ValueError(f"Evaluation row {index + 1} has invalid claim annotations.")
    return data


def save_dataset(path: str | Path, dataset: Sequence[dict]) -> None:
    normalized = []
    for row in dataset:
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError("Every evaluation row must include a question.")
        relevant_ids = row.get("relevant_chunk_ids", [])
        if isinstance(relevant_ids, str):
            relevant_ids = [item.strip() for item in relevant_ids.split(",") if item.strip()]
        if not isinstance(relevant_ids, list):
            raise ValueError(f"Evaluation question {row['question']!r} has invalid relevant IDs.")
        claims_total = int(row.get("claims_total") or 0)
        claims_supported = int(row.get("claims_supported") or 0)
        if claims_total < 0 or claims_supported < 0 or claims_supported > claims_total:
            raise ValueError(
                "Supported claim labels must be between zero and the total claim count."
            )
        normalized.append(
            {
                "question": question,
                "expected_answer": str(row.get("expected_answer", "")).strip(),
                "relevant_chunk_ids": list(relevant_ids),
                "claims_total": claims_total,
                "claims_supported": claims_supported,
            }
        )
    Path(path).write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_retrieval_dataset(
    dataset: Sequence[dict],
    retrieve_ids: Callable[[str, int], Sequence[str]],
    *,
    k: int,
) -> dict:
    rows = []
    retrieval_scores = []
    for row in dataset:
        relevant_ids = row.get("relevant_chunk_ids", [])
        if isinstance(relevant_ids, str):
            relevant_ids = [item.strip() for item in relevant_ids.split(",") if item.strip()]
        if not relevant_ids:
            continue
        retrieved_ids = list(retrieve_ids(row["question"], k))
        scores = retrieval_metrics(retrieved_ids, relevant_ids, k=k)
        retrieval_scores.append(scores)
        rows.append(
            {
                "question": row["question"],
                "expected_answer": row.get("expected_answer", ""),
                "relevant_chunk_ids": list(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "claims_total": int(row.get("claims_total") or 0),
                "claims_supported": int(row.get("claims_supported") or 0),
                "groundedness_score": (
                    int(row.get("claims_supported") or 0)
                    / int(row.get("claims_total") or 0)
                    if int(row.get("claims_total") or 0) > 0
                    else None
                ),
                **scores,
            }
        )
    groundedness = aggregate_groundedness(dataset)
    return {
        "evaluated_questions": len(rows),
        "rows": rows,
        **aggregate_retrieval_metrics(retrieval_scores),
        "groundedness_score": groundedness,
        "groundedness_method": (
            "Manual annotation: supported factual claims divided by all annotated factual claims."
        ),
    }
