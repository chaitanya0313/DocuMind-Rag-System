import json

import pytest

from evaluation import evaluate_retrieval_dataset, load_dataset, save_dataset
from metrics import (
    MetricsStore,
    average_measurement,
    aggregate_groundedness,
    aggregate_retrieval_metrics,
    estimate_tokens,
    new_query_record,
    query_summary,
    retrieval_metrics,
)


def test_precision_recall_and_mrr_use_labeled_chunk_ids():
    scores = retrieval_metrics(["a", "x", "b"], ["a", "b"], k=3)
    assert scores["precision_at_k"] == pytest.approx(2 / 3)
    assert scores["recall_at_k"] == 1.0
    assert scores["mrr"] == 1.0

    missed = retrieval_metrics(["x", "y"], ["a"], k=2)
    assert missed == {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    assert retrieval_metrics(["a"], [], k=4)["precision_at_k"] is None
    assert aggregate_retrieval_metrics([scores, missed])["mrr"] == 0.5


def test_evaluation_excludes_unlabeled_rows(tmp_path):
    dataset = [
        {"question": "q1", "relevant_chunk_ids": ["a"]},
        {"question": "q2", "relevant_chunk_ids": []},
    ]
    result = evaluate_retrieval_dataset(
        dataset, lambda question, k: ["a"], k=1
    )
    assert result["evaluated_questions"] == 1
    assert result["precision_at_k"] == 1.0
    assert result["groundedness_score"] is None


def test_evaluation_dataset_round_trip_and_manual_labels(tmp_path):
    path = tmp_path / "evaluation.json"
    dataset = [
        {
            "question": "q",
            "expected_answer": "a",
            "relevant_chunk_ids": "chunk1, chunk2",
            "claims_total": 2,
            "claims_supported": 1,
        }
    ]
    save_dataset(path, dataset)
    loaded = load_dataset(path)
    assert loaded[0]["relevant_chunk_ids"] == ["chunk1", "chunk2"]
    assert aggregate_groundedness(loaded) == 0.5
    with pytest.raises(ValueError, match="Supported claim labels"):
        save_dataset(
            path,
            [
                {
                    "question": "q",
                    "relevant_chunk_ids": [],
                    "claims_total": 1,
                    "claims_supported": 2,
                }
            ],
        )


def test_token_estimation_and_measurement_persistence(tmp_path):
    assert estimate_tokens("") == 0
    assert estimate_tokens("A short sentence for token estimation.") > 0
    metrics_store = MetricsStore(tmp_path / "metrics.sqlite3")
    record = new_query_record()
    record.update({"question": "What is the answer?", "answer_status": "success"})
    record_id = metrics_store.record(record)
    record["citation_support_score"] = 0.75
    metrics_store.update(record_id, record)
    recent = metrics_store.recent()
    assert recent[0]["citation_support_score"] == 0.75
    assert query_summary(recent) == {
        "queries_tested": 1,
        "successful_answers": 1,
        "insufficient_context_responses": 0,
        "api_failures": 0,
        "benchmark_questions": 0,
    }
    assert query_summary(
        [{"answer_status": "retrieval_evaluation"}]
    ) == {
        "queries_tested": 0,
        "successful_answers": 0,
        "insufficient_context_responses": 0,
        "api_failures": 0,
        "benchmark_questions": 1,
    }
    assert average_measurement(
        [
            {"generation_latency_ms": 100},
            {"generation_latency_ms": None},
            {"generation_latency_ms": 300},
        ],
        "generation_latency_ms",
    ) == 200
    assert average_measurement([{"retrieval_latency_ms": None}], "retrieval_latency_ms") is None


def test_invalid_dataset_shape_is_reported(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        load_dataset(path)
