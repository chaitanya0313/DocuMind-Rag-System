"""Document extraction and duplicate-aware vector indexing."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass

from chunking import chunk_pages
from schemas import SourcePage, TextChunk

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass(frozen=True)
class IngestionReport:
    uploaded_documents: int
    pages_extracted: int
    chunks_created: int
    duplicates_skipped: int
    duration_ms: float
    errors: tuple[str, ...]


def extract_document(filename: str, content: bytes) -> list[SourcePage]:
    suffix = filename.lower().rsplit(".", maxsplit=1)[-1] if "." in filename else ""
    extension = f".{suffix}"
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")
    if not content:
        raise ValueError("The uploaded file is empty.")

    if extension == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Install PyMuPDF to read PDF files.") from exc
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                pages = [
                    SourcePage(text=page.get_text("text"), page_number=index)
                    for index, page in enumerate(pdf, start=1)
                ]
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
        if not any(page.text.strip() for page in pages):
            raise ValueError("The PDF contains no extractable text.")
        return pages

    if extension == ".docx":
        try:
            from docx import Document
            from docx.opc.exceptions import PackageNotFoundError
            from lxml.etree import XMLSyntaxError
        except ImportError as exc:
            raise RuntimeError("Install python-docx to read DOCX files.") from exc
        try:
            document = Document(io.BytesIO(content))
        except (
            PackageNotFoundError,
            XMLSyntaxError,
            ValueError,
            OSError,
            zipfile.BadZipFile,
        ) as exc:
            raise ValueError(f"Could not read DOCX: {exc}") from exc
        blocks = [paragraph.text for paragraph in document.paragraphs]
        blocks.extend(
            cell.text
            for table in document.tables
            for row in table.rows
            for cell in row.cells
        )
        text = "\n".join(block for block in blocks if block.strip())
        if not text.strip():
            raise ValueError("The DOCX contains no extractable text.")
        return [SourcePage(text=text, page_number=None)]

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The TXT file is not valid UTF-8 text.") from exc
    if not text.strip():
        raise ValueError("The TXT file contains no text.")
    return [SourcePage(text=text, page_number=None)]


def document_id_for_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def ingest_documents(
    files: Iterable[tuple[str, bytes]],
    *,
    store: object,
    embedder: object,
    chunk_size: int = 800,
    overlap: int = 120,
) -> IngestionReport:
    started = time.perf_counter()
    uploaded = 0
    pages_extracted = 0
    chunks_created = 0
    duplicates = 0
    errors: list[str] = []

    for filename, content in files:
        if not content:
            errors.append(f"{filename}: The uploaded file is empty.")
            continue
        document_id = document_id_for_content(content)
        try:
            pages = extract_document(filename, content)
            pages_extracted += sum(page.page_number is not None for page in pages)
            if store.has_document(document_id):
                duplicates += 1
                continue
            chunks = chunk_pages(
                pages,
                document_id=document_id,
                filename=filename,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if not chunks:
                errors.append(f"{filename}: No text chunks could be created.")
                continue
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            store.add_chunks(chunks, vectors)
            uploaded += 1
            chunks_created += len(chunks)
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append(f"{filename}: {exc}")

    duration_ms = (time.perf_counter() - started) * 1000
    return IngestionReport(
        uploaded_documents=uploaded,
        pages_extracted=pages_extracted,
        chunks_created=chunks_created,
        duplicates_skipped=duplicates,
        duration_ms=duration_ms,
        errors=tuple(errors),
    )
