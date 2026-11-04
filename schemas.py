"""Small data types shared by the application modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourcePage:
    text: str
    page_number: int | None


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    content_hash: str
    text: str
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    content_hash: str
    text: str
    distance: float
    citation_id: str
    chunk_size: int = 800
    chunk_overlap: int = 120
