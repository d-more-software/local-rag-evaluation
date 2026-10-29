
from langchain_core.documents import Document

from rag_evaluation.config import settings
from rag_evaluation.retrieval.embeddings import create_embeddings
from rag_evaluation.retrieval.vector_store import load_vector_store


def search_documents(
    question: str,
    top_k: int | None = None,
) -> list[Document]:
    """Retrieve the most relevant document chunks for a question."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    embeddings = create_embeddings()
    vector_store = load_vector_store(embeddings)

    return vector_store.similarity_search(
        question,
        k=top_k or settings.top_k,
    )
