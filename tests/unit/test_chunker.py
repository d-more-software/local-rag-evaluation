import pytest
from langchain_core.documents import Document

from rag_evaluation.ingestion.chunker import chunk_documents


def test_chunk_documents_preserves_metadata():
    document = Document(
        page_content=" ".join(["RAG retrieval generation"] * 100),
        metadata={"page": 0, "source": "test.pdf"},
    )

    chunks = chunk_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_chunk_documents_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_documents(
            [Document(page_content="Test")],
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_documents_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one document"):
        chunk_documents([])
