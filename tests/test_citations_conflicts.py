from citations import (
    evaluate_citation_support,
    extract_answer_claims,
    format_document_citations,
    is_refusal_answer,
    validate_citations,
)
from conflict_detection import detect_conflicts
from schemas import RetrievedChunk


def source(citation_id, filename, text, chunk_id):
    return RetrievedChunk(
        document_id=chunk_id,
        filename=filename,
        page_number=None,
        chunk_id=chunk_id,
        content_hash=chunk_id,
        text=text,
        distance=0.1,
        citation_id=citation_id,
    )


def test_citation_validation_keeps_known_and_removes_unknown_ids():
    answer, valid, invalid = validate_citations(
        "The library opens at 8 AM [S1]. An unsupported detail [S9].",
        {"S1", "S2"},
    )
    assert answer == "The library opens at 8 AM [S1]. An unsupported detail."
    assert valid == ["S1"]
    assert invalid == ["S9"]


def test_document_citations_map_to_ordered_source_ids_and_keep_unmatched_references():
    answer, valid, invalid = validate_citations(
        "First source [Doc 1]. Second source [Doc 2]. Missing source [Doc 3].",
        {"S1", "S2"},
    )
    assert answer == "First source [Doc 1]. Second source [Doc 2]. Missing source [Doc 3]."
    assert valid == ["S1", "S2"]
    assert invalid == ["Doc 3"]
    assert extract_answer_claims("The library opens at eight [Doc 1].")[0]["citation_ids"] == [
        "S1"
    ]


def test_document_citations_format_as_source_links_or_missing_placeholders():
    formatted, missing = format_document_citations(
        "The library opens at eight [Doc 1]. This detail is missing [Doc 3].",
        [
            {"citation_id": "S1", "filename": "campus.pdf", "page_number": 2},
            {"citation_id": "S2", "filename": "guide.txt", "page_number": None},
        ],
    )
    assert formatted == (
        "The library opens at eight [S1: campus\\.pdf p\\.2](#source-S1). "
        "This detail is missing [Doc 3: source not found]."
    )
    assert missing == [3]


def test_refusal_answer_does_not_require_a_citation():
    assert is_refusal_answer(
        "The provided documents do not say what the late-return fine is."
    )


def test_citation_support_is_not_the_same_as_id_validity():
    claims = extract_answer_claims("The library opens at eight [S1]. The lab closes at five [S2].")
    result = evaluate_citation_support(claims, {"0": True, "1": False})
    assert result == {
        "evaluated_claims": 2,
        "supported_claims": 1,
        "unsupported_claims": 1,
        "score": 0.5,
    }
    assert evaluate_citation_support(claims, {"0": None, "1": None})["score"] is None


def test_conflict_detection_finds_disagreeing_times_and_names_sources():
    conflicts = detect_conflicts(
        [
            source(
                "S1",
                "campus.txt",
                "The library is open from 8:00 AM to 8:00 PM on weekdays.",
                "chunk-one",
            ),
            source(
                "S2",
                "handbook.txt",
                "The library is open from 9:00 AM to 7:00 PM on weekdays.",
                "chunk-two",
            ),
        ]
    )
    assert len(conflicts) == 1
    assert conflicts[0].first.filename == "campus.txt"
    assert conflicts[0].second.filename == "handbook.txt"


def test_conflict_detector_does_not_flag_unrelated_passages():
    assert detect_conflicts(
        [
            source("S1", "a.txt", "The library has a reading room.", "one"),
            source("S2", "b.txt", "The science lab uses safety glasses.", "two"),
        ]
    ) == []


def test_conflict_detector_does_not_compare_chunks_from_the_same_document():
    assert detect_conflicts(
        [
            source(
                "S1",
                "campus.pdf",
                "Library opens at 8 AM.",
                "page-one",
            ),
            source(
                "S2",
                "campus.pdf",
                "Students may borrow 5 books for 14 days.",
                "page-two",
            ),
        ]
    ) == []


def test_conflict_detector_matches_the_same_topic_and_unit_in_demo_text():
    conflicts = detect_conflicts(
        [
            source(
                "S1",
                "services.txt",
                "CAMPUS SHUTTLE\nThe campus shuttle departs every 20 minutes between the main gate and the library.",
                "services",
            ),
            source(
                "S2",
                "transport.txt",
                "WEEKDAY CAMPUS SHUTTLE\nThe transport notice says the campus shuttle runs every 30 minutes between the main gate and library.",
                "transport",
            ),
        ]
    )
    assert len(conflicts) == 1
    assert "frequenc" in conflicts[0].reason


def test_conflict_detector_does_not_mix_library_hours_with_shuttle_hours():
    assert detect_conflicts(
        [
            source(
                "S1",
                "services.txt",
                "LIBRARY HOURS\nThe campus library is open from 8:00 AM to 8:00 PM.",
                "hours",
            ),
            source(
                "S2",
                "transport.txt",
                "WEEKDAY SHUTTLE\nThe shuttle runs every 20 minutes from 9:00 AM to 6:00 PM.",
                "shuttle",
            ),
        ]
    ) == []
