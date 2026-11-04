from __future__ import annotations

from pathlib import Path

from chunking import chunk_pages
from evaluation import load_dataset
from ingestion import document_id_for_content, extract_document
from load_demo import DEMO_FILENAMES


def test_demo_documents_are_uploadable_and_match_seeded_evaluation_labels():
    sample_dir = Path(__file__).resolve().parents[1] / "sample_docs"
    available_chunk_ids = set()

    assert len(DEMO_FILENAMES) == 4
    for filename in DEMO_FILENAMES:
        content = (sample_dir / filename).read_bytes()
        pages = extract_document(filename, content)
        chunks = chunk_pages(
            pages,
            document_id=document_id_for_content(content),
            filename=filename,
        )
        assert len(chunks) == 1
        available_chunk_ids.add(chunks[0].chunk_id)

    dataset = load_dataset(Path(__file__).resolve().parents[1] / "evaluation_dataset.json")
    assert len(dataset) == 6
    for row in dataset:
        assert row["relevant_chunk_ids"]
        assert set(row["relevant_chunk_ids"]).issubset(available_chunk_ids)


def test_demo_question_file_includes_conflict_and_abstention_prompts():
    questions = (Path(__file__).resolve().parents[1] / "sample_docs" / "DEMO_QUESTIONS.txt").read_text(
        encoding="utf-8"
    )
    assert "every 20 minutes" in questions
    assert "every 30 minutes" in questions
    assert "favorite color" in questions
    assert "sufficient supporting information" in questions
