from __future__ import annotations

import io

import fitz
import pytest
from docx import Document

from ingestion import extract_document, ingest_documents


def make_pdf(*page_texts: str) -> bytes:
    document = fitz.open()
    for page_text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), page_text)
    return document.tobytes()


def make_docx(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_pdf_extraction_preserves_page_numbers():
    pages = extract_document("guide.pdf", make_pdf("First page text", "Second page text"))
    assert [(page.page_number, page.text.strip()) for page in pages] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_txt_extraction_has_no_invented_page_number():
    pages = extract_document("notes.txt", b"Useful plain text")
    assert len(pages) == 1
    assert pages[0].text == "Useful plain text"
    assert pages[0].page_number is None


def test_docx_extraction_includes_paragraphs_and_tables():
    document = Document()
    document.add_paragraph("DOCX paragraph")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Table content"
    buffer = io.BytesIO()
    document.save(buffer)

    page = extract_document("notes.docx", buffer.getvalue())[0]
    assert "DOCX paragraph" in page.text
    assert "Table content" in page.text
    assert page.page_number is None


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("data.csv", b"content", "Unsupported"),
        ("empty.txt", b"", "empty"),
        ("blank.txt", b" \n", "no text"),
        ("broken.txt", b"\xff", "UTF-8"),
    ],
)
def test_invalid_or_empty_documents_are_reported(filename, content, message):
    with pytest.raises(ValueError, match=message):
        extract_document(filename, content)


def test_duplicate_document_is_not_embedded_twice(fake_store, fake_embedder):
    file = ("guide.txt", b"A short campus guide.")
    first = ingest_documents([file], store=fake_store, embedder=fake_embedder)
    second = ingest_documents([file], store=fake_store, embedder=fake_embedder)

    assert first.uploaded_documents == 1
    assert first.chunks_created == 1
    assert second.uploaded_documents == 0
    assert second.duplicates_skipped == 1
    assert len(fake_store.rows) == 1
    assert first.pages_extracted == 0


def test_ingestion_counts_pdf_pages_but_does_not_invent_text_file_pages(
    fake_store, fake_embedder
):
    report = ingest_documents(
        [
            ("guide.pdf", make_pdf("Page one", "Page two")),
            ("notes.txt", b"A text file does not have page numbers."),
        ],
        store=fake_store,
        embedder=fake_embedder,
    )
    assert report.uploaded_documents == 2
    assert report.pages_extracted == 2


def test_reset_removes_chunks_and_prevents_stale_retrieval(fake_store, fake_embedder):
    ingest_documents(
        [("guide.txt", b"A short campus guide.")],
        store=fake_store,
        embedder=fake_embedder,
    )
    assert fake_store.document_count() == 1
    fake_store.reset()
    assert fake_store.document_count() == 0
    assert fake_store.search([1.0, 1.0], 4) == []
