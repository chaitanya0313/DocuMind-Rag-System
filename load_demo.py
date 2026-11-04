"""Ingest the four bundled TXT demo documents into the active Chroma collection."""

from __future__ import annotations

from config import BASE_DIR
from embeddings import SentenceTransformerEmbedder
from ingestion import ingest_documents
from vector_store import ChromaVectorStore

DEMO_FILENAMES = (
    "demo_campus_services.txt",
    "demo_student_handbook.txt",
    "demo_transport_notice.txt",
    "demo_research_lab.txt",
)


def main() -> int:
    paths = [BASE_DIR / "sample_docs" / filename for filename in DEMO_FILENAMES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Demo documents are missing: {', '.join(missing)}")

    store = ChromaVectorStore()
    embedder = SentenceTransformerEmbedder()
    report = ingest_documents(
        ((path.name, path.read_bytes()) for path in paths),
        store=store,
        embedder=embedder,
    )
    print(
        f"Added {report.uploaded_documents} document(s), created "
        f"{report.chunks_created} chunk(s), skipped {report.duplicates_skipped} "
        f"duplicate(s) in {report.duration_ms:.1f} ms."
    )
    for error in report.errors:
        print(f"Error: {error}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
