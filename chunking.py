"""Character-based chunking; token counts are estimated separately."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas import SourcePage, TextChunk


def chunk_pages(
    pages: Iterable[SourcePage],
    *,
    document_id: str,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[TextChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[""],
        keep_separator=False,
        strip_whitespace=False,
        add_start_index=True,
    )
    chunks: list[TextChunk] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        for page_chunk_index, split in enumerate(splitter.create_documents([text])):
            content = split.page_content
            page_label = str(page.page_number) if page.page_number is not None else "na"
            chunk_id = f"{document_id[:12]}-p{page_label}-c{page_chunk_index:04d}"
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_id=chunk_id,
                    content_hash=content_hash,
                    text=content,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
            )
    return chunks
