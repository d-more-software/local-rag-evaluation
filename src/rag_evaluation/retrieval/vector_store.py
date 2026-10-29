from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_evaluation.config import settings


def create_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Build a FAISS index from document chunks."""
    if not documents:
        raise ValueError("At least one document chunk is required.")

    return FAISS.from_documents(documents, embeddings)


def save_vector_store(
    vector_store: FAISS,
    index_dir: Path | None = None,
) -> None:
    """Persist the FAISS index locally."""
    destination = index_dir or settings.faiss_index_dir
    destination.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(destination))


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    index_dir: Path | None = None,
) -> FAISS:
    """Load a previously saved FAISS index."""
    source = index_dir or settings.faiss_index_dir

    if not (source / "index.faiss").is_file():
        raise FileNotFoundError(f"FAISS index not found in {source}")

    return FAISS.load_local(
        str(source),
        embeddings,
        allow_dangerous_deserialization=True,
    )
