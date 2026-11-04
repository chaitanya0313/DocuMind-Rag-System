"""Persistent ChromaDB wrapper for document chunks."""

from __future__ import annotations

from collections.abc import Sequence

from config import CHROMA_DIR
from schemas import TextChunk


class ChromaVectorStore:
    COLLECTION_NAME = "documind_documents"

    def __init__(self, persist_directory=CHROMA_DIR) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("Install ChromaDB to use persistent document storage.") from exc
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: Sequence[TextChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number if chunk.page_number is not None else -1,
                    "chunk_id": chunk.chunk_id,
                    "content_hash": chunk.content_hash,
                    "chunk_size": chunk.chunk_size,
                    "chunk_overlap": chunk.chunk_overlap,
                }
                for chunk in chunks
            ],
        )

    def has_document(self, document_id: str) -> bool:
        return bool(
            self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )["ids"]
        )

    def search(self, embedding: Sequence[float], top_k: int) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if not self._collection.count():
            return []
        response = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict] = []
        for chunk_id, text, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            rows.append(
                {
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "page_number": metadata["page_number"],
                    "chunk_id": chunk_id,
                    "content_hash": metadata["content_hash"],
                    "chunk_size": metadata.get("chunk_size", 800),
                    "chunk_overlap": metadata.get("chunk_overlap", 120),
                    "text": text,
                    "distance": float(distance),
                }
            )
        return rows

    def document_count(self) -> int:
        response = self._collection.get(include=["metadatas"])
        return len({metadata["document_id"] for metadata in response["metadatas"]})

    def indexed_files(self) -> list[dict]:
        response = self._collection.get(include=["metadatas"])
        files: dict[str, dict] = {}
        for metadata in response["metadatas"]:
            filename = metadata["filename"]
            file = files.setdefault(
                filename,
                {"filename": filename, "chunk_count": 0, "pages": set()},
            )
            file["chunk_count"] += 1
            page_number = metadata.get("page_number", -1)
            if page_number > 0:
                file["pages"].add(page_number)
        return [
            {
                "filename": file["filename"],
                "chunk_count": file["chunk_count"],
                "page_count": len(file["pages"]),
            }
            for file in sorted(files.values(), key=lambda item: item["filename"])
        ]

    def delete_filename(self, filename: str) -> int:
        if not filename:
            raise ValueError("filename must not be empty")
        ids = self._collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def chunk_count(self) -> int:
        return self._collection.count()

    def all_chunks(self) -> list[dict]:
        response = self._collection.get(
            include=["documents", "metadatas"],
        )
        rows = []
        for chunk_id, text, metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        ):
            rows.append(
                {
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "page_number": metadata["page_number"],
                    "chunk_id": chunk_id,
                    "content_hash": metadata["content_hash"],
                    "chunk_size": metadata.get("chunk_size", 800),
                    "chunk_overlap": metadata.get("chunk_overlap", 120),
                    "text": text,
                }
            )
        return rows

    def reset(self) -> None:
        self._client.delete_collection(name=self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
