
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel

from rag_evaluation.config import settings


def create_judge() -> OllamaModel:
    """Create the local LLM judge."""
    return OllamaModel(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def create_metrics() -> list:
    """Create the RAG evaluation metrics."""
    judge = create_judge()

    return [
        FaithfulnessMetric(
            threshold=0.7,
            model=judge,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=0.7,
            model=judge,
            include_reason=True,
        ),
    ]
