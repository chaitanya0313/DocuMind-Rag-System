from __future__ import annotations

import hashlib

import pytest


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class FakeStore:
    def __init__(self):
        self.rows = {}

    def has_document(self, document_id):
        return any(row["document_id"] == document_id for row in self.rows.values())

    def add_chunks(self, chunks, vectors):
        for chunk, vector in zip(chunks, vectors):
            self.rows[chunk.chunk_id] = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "chunk_size": chunk.chunk_size,
                "chunk_overlap": chunk.chunk_overlap,
                "text": chunk.text,
                "embedding": vector,
            }

    def search(self, embedding, top_k):
        return [
            {
                key: row[key]
                for key in (
                    "document_id",
                    "filename",
                    "page_number",
                    "chunk_id",
                    "content_hash",
                    "chunk_size",
                    "chunk_overlap",
                    "text",
                )
            }
            | {"distance": 0.1}
            for row in list(self.rows.values())[:top_k]
        ]

    def document_count(self):
        return len({row["document_id"] for row in self.rows.values()})

    def indexed_files(self):
        files = {}
        for row in self.rows.values():
            file = files.setdefault(
                row["filename"],
                {"filename": row["filename"], "chunk_count": 0, "pages": set()},
            )
            file["chunk_count"] += 1
            if row["page_number"] > 0:
                file["pages"].add(row["page_number"])
        return [
            {
                "filename": file["filename"],
                "chunk_count": file["chunk_count"],
                "page_count": len(file["pages"]),
            }
            for file in sorted(files.values(), key=lambda item: item["filename"])
        ]

    def delete_filename(self, filename):
        matching_ids = [
            chunk_id
            for chunk_id, row in self.rows.items()
            if row["filename"] == filename
        ]
        for chunk_id in matching_ids:
            del self.rows[chunk_id]
        return len(matching_ids)

    def reset(self):
        self.rows.clear()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_store():
    return FakeStore()


def chunk_id_for_text(filename: str, content: bytes) -> str:
    document_id = hashlib.sha256(content).hexdigest()
    return f"{document_id[:12]}-pna-c0000"
