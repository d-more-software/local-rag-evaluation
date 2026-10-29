
import unicodedata
from typing import TypedDict


class CompletenessResult(TypedDict):
    score: float
    found: list[str]
    missing: list[str]


def normalize_text(text: str) -> str:
    """Normalize text for simple fact matching."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )


def check_completeness(
    answer: str,
    required_facts: list[str],
) -> CompletenessResult:
    """Measure the proportion of required facts found in an answer."""
    normalized_answer = normalize_text(answer)

    found = [
        fact
        for fact in required_facts
        if normalize_text(fact) in normalized_answer
    ]

    missing = [
        fact
        for fact in required_facts
        if normalize_text(fact) not in normalized_answer
    ]

    total = len(required_facts)

    return {
        "score": len(found) / total if total else 1.0,
        "found": found,
        "missing": missing,
    }
