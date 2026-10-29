from langchain_huggingface import HuggingFaceEmbeddings

from rag_evaluation.config import settings


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used by the RAG pipeline."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
