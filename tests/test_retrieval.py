import pytest

from ingestion import ingest_documents
from retrieval import has_sufficient_relevance, retrieve
from schemas import TextChunk
from vector_store import ChromaVectorStore


def test_retrieval_maps_sources_and_orders_by_distance(fake_store, fake_embedder):
    ingest_documents(
        [
            ("first.txt", b"The campus library opens at eight in the morning."),
            ("second.txt", b"The science laboratory requires safety glasses."),
        ],
        store=fake_store,
        embedder=fake_embedder,
    )
    results, latency = retrieve(
        "When does the campus library open?",
        store=fake_store,
        embedder=fake_embedder,
        top_k=2,
    )

    assert [source.citation_id for source in results] == ["S1", "S2"]
    assert all(source.filename in {"first.txt", "second.txt"} for source in results)
    assert all(source.page_number is None for source in results)
    assert latency >= 0


def test_retrieval_defaults_to_top_four(fake_store, fake_embedder):
    requested_top_k = []
    original_search = fake_store.search

    def capture_search(embedding, top_k):
        requested_top_k.append(top_k)
        return original_search(embedding, top_k)

    fake_store.search = capture_search
    retrieve("question", store=fake_store, embedder=fake_embedder)

    assert requested_top_k == [4]


def test_relevance_uses_raw_distance_threshold_not_percentage():
    from schemas import RetrievedChunk

    source = RetrievedChunk(
        "doc",
        "file.txt",
        None,
        "chunk",
        "hash",
        "text",
        0.4,
        "S1",
    )
    assert has_sufficient_relevance([source], max_distance=0.7)
    assert not has_sufficient_relevance([source], max_distance=0.3)
    assert not has_sufficient_relevance([])


def test_empty_question_and_invalid_top_k_are_rejected(fake_store, fake_embedder):
    with pytest.raises(ValueError):
        retrieve("  ", store=fake_store, embedder=fake_embedder)
    with pytest.raises(ValueError):
        retrieve("question", store=fake_store, embedder=fake_embedder, top_k=0)


def test_persistent_chroma_search_metadata_and_reset(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma")
    chunk = TextChunk(
        document_id="a" * 64,
        filename="guide.pdf",
        page_number=2,
        chunk_id="aaaaaaaaaaaa-p2-c0000",
        content_hash="b" * 64,
        text="The library opens in the morning.",
    )
    store.add_chunks([chunk], [[1.0, 0.0]])

    assert store.document_count() == 1
    assert store.chunk_count() == 1
    assert store.has_document("a" * 64)
    row = store.search([1.0, 0.0], 4)[0]
    assert row["filename"] == "guide.pdf"
    assert row["page_number"] == 2
    assert row["chunk_id"] == chunk.chunk_id
    assert row["text"] == chunk.text

    store.reset()
    assert store.document_count() == 0
    assert store.search([1.0, 0.0], 4) == []


def test_persistent_chroma_lists_and_deletes_only_matching_filename(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma")
    chunks = [
        TextChunk("a" * 64, "guide.pdf", 1, "guide-p1", "b" * 64, "Page one"),
        TextChunk("a" * 64, "guide.pdf", 2, "guide-p2", "c" * 64, "Page two"),
        TextChunk("d" * 64, "notes.txt", None, "notes-pna", "e" * 64, "Notes"),
    ]
    store.add_chunks(chunks, [[1.0, 0.0]] * len(chunks))

    assert store.indexed_files() == [
        {"filename": "guide.pdf", "chunk_count": 2, "page_count": 2},
        {"filename": "notes.txt", "chunk_count": 1, "page_count": 0},
    ]
    assert store.delete_filename("guide.pdf") == 2
    assert store.chunk_count() == 1
    assert store.indexed_files() == [
        {"filename": "notes.txt", "chunk_count": 1, "page_count": 0}
    ]
