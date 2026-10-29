
from langchain_ollama import ChatOllama

from rag_evaluation.config import settings


def create_llm() -> ChatOllama:
    """Create the local language model."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
