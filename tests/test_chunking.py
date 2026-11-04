import pytest

from chunking import chunk_pages
from schemas import SourcePage


def test_chunking_applies_character_overlap_and_stable_metadata():
    pages = [SourcePage(text="0123456789abcdefghij", page_number=3)]
    chunks = chunk_pages(
        pages,
        document_id="a" * 64,
        filename="report.pdf",
        chunk_size=10,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == ["0123456789", "789abcdefg", "efghij"]
    assert [chunk.page_number for chunk in chunks] == [3, 3, 3]
    assert [chunk.chunk_id for chunk in chunks] == [
        "aaaaaaaaaaaa-p3-c0000",
        "aaaaaaaaaaaa-p3-c0001",
        "aaaaaaaaaaaa-p3-c0002",
    ]
    assert all(chunk.filename == "report.pdf" for chunk in chunks)
    assert all(len(chunk.content_hash) == 64 for chunk in chunks)
    assert all(chunk.chunk_size == 10 and chunk.chunk_overlap == 3 for chunk in chunks)


def test_unpaged_documents_do_not_get_a_page_number():
    chunk = chunk_pages(
        [SourcePage("word document", None)],
        document_id="b" * 64,
        filename="memo.docx",
    )[0]
    assert chunk.page_number is None
    assert "-pna-" in chunk.chunk_id


@pytest.mark.parametrize(("chunk_size", "overlap"), [(0, 0), (10, -1), (10, 10)])
def test_invalid_chunk_configuration_fails(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            [SourcePage("text", None)],
            document_id="c" * 64,
            filename="file.txt",
            chunk_size=chunk_size,
            overlap=overlap,
        )
