
import json
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from rag_evaluation.evaluation.completeness import check_completeness
from rag_evaluation.evaluation.evaluator import create_metrics
from rag_evaluation.generation.answer import generate_answer

DATASET_PATH = Path("data/evaluation/golden_dataset.json")


def load_dataset() -> list[dict]:
    """Load the evaluation dataset."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def run_evaluation() -> list[dict]:
    """Generate answers and collect completeness results."""
    dataset = load_dataset()
    results = []
    test_cases = []

    for item in dataset:
        question = item["question"]
        answer, documents = generate_answer(question)

        completeness = check_completeness(
            answer,
            item["required_facts"],
        )

        retrieval_context = [
            document.page_content for document in documents
        ]

        results.append(
            {
                "question": question,
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "completeness_score": completeness["score"],
                "found_facts": completeness["found"],
                "missing_facts": completeness["missing"],
                "sources": retrieval_context,
            }
        )

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=item["expected_answer"],
                retrieval_context=retrieval_context,
            )
        )

    metrics = create_metrics()

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    return results


def main() -> None:
    results = run_evaluation()

    print("\n=== RÉSUMÉ DE COMPLÉTUDE ===")

    for result in results:
        print(f"\nQuestion : {result['question']}")
        print(f"Complétude : {result['completeness_score']:.0%}")
        print(f"Faits manquants : {result['missing_facts']}")

    if results:
        average = sum(
            result["completeness_score"] for result in results
        ) / len(results)

        print(f"\nComplétude moyenne : {average:.0%}")


if __name__ == "__main__":
    main()
