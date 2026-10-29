
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_evaluation.config import settings
from rag_evaluation.retrieval.embeddings import create_embeddings
from rag_evaluation.retrieval.vector_store import load_vector_store


def hybrid_search(
    question: str,
    top_k: int | None = None,
) -> list[Document]:
    """Combine semantic FAISS search and lexical BM25 search."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    k = top_k or settings.top_k
    embeddings = create_embeddings()
    vector_store = load_vector_store(embeddings)

    # Retrieve semantic matches from FAISS.
    semantic_docs = vector_store.similarity_search(question, k=k)

    # Retrieve lexical matches from the indexed chunks.
    all_docs = list(vector_store.docstore._dict.values())

    if not all_docs:
        return semantic_docs

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k
    lexical_docs = bm25.invoke(question)

    # Reciprocal Rank Fusion (RRF): combine both rankings.
    scores = {}
    documents_by_key = {}

    for results in (semantic_docs, lexical_docs):
        for rank, document in enumerate(results, start=1):
            key = (
                document.metadata.get("source", ""),
                document.metadata.get("page", ""),
                document.page_content,
            )
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
            documents_by_key[key] = document

    ranked_keys = sorted(scores, key=scores.get, reverse=True)

    return [documents_by_key[key] for key in ranked_keys[:k]]
