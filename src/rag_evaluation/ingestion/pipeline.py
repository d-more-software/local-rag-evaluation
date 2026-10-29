
from pathlib import Path

from rag_evaluation.config import settings
from rag_evaluation.ingestion.chunker import chunk_documents
from rag_evaluation.ingestion.loader import load_pdf
from rag_evaluation.retrieval.embeddings import create_embeddings
from rag_evaluation.retrieval.vector_store import (
    create_vector_store,
    save_vector_store,
)


def index_documents(
    documents_dir: Path | None = None,
) -> int:
    """Load PDFs, split them, build a FAISS index, and save it."""
    source_dir = documents_dir or settings.documents_dir

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {source_dir}"
        )

    pdf_paths = sorted(source_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {source_dir}"
        )

    documents = []

    for pdf_path in pdf_paths:
        documents.extend(load_pdf(pdf_path))

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    save_vector_store(vector_store)

    return len(chunks)
