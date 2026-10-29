import pytest

from rag_evaluation.retrieval.vector_store import create_vector_store


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_create_vector_store_rejects_empty_documents():
    with pytest.raises(ValueError, match="At least one document chunk"):
        create_vector_store([], FakeEmbeddings())
