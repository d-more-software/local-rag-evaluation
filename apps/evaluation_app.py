
import json
from pathlib import Path

import gradio as gr
import pandas as pd

from rag_evaluation.evaluation.completeness import check_completeness
from rag_evaluation.generation.answer import generate_answer

DATASET_PATH = Path("data/evaluation/golden_dataset.json")


def load_dataset() -> list[dict]:
    """Load the evaluation dataset."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_question(question: str) -> tuple[str, str]:
    """Answer a question and display the retrieved context."""
    if not question or not question.strip():
        return "Saisis une question.", ""

    answer, documents = generate_answer(question.strip())

    sources = "\n\n".join(
        f"Source {index} :\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )

    return answer, sources


def evaluate_dataset() -> tuple[pd.DataFrame, str]:
    """Evaluate dataset completeness and display the results."""
    dataset = load_dataset()
    results = []

    for item in dataset:
        answer, documents = generate_answer(item["question"])

        completeness = check_completeness(
            answer,
            item["required_facts"],
        )

        results.append(
            {
                "Question": item["question"],
                "Réponse": answer,
                "Complétude": f"{completeness['score']:.0%}",
                "Faits manquants": ", ".join(completeness["missing"]),
                "Sources récupérées": len(documents),
            }
        )

    dataframe = pd.DataFrame(results)

    average = (
        sum(
            result["completeness_score"]
            for result in [
                {
                    "completeness_score": check_completeness(
                        row["Réponse"],
                        next(
                            item["required_facts"]
                            for item in dataset
                            if item["question"] == row["Question"]
                        ),
                    )["score"],
                }
                for row in results
            ]
        )
        / len(results)
        if results
        else 0.0
    )

    return dataframe, f"Complétude moyenne : {average:.0%}"


with gr.Blocks(title="RAG Evaluation") as demo:
    gr.Markdown("# RAG Evaluation")
    gr.Markdown(
        "Teste le système RAG, consulte les sources "
        "et évalue la complétude des réponses."
    )

    with gr.Tab("Question manuelle"):
        question_input = gr.Textbox(
            label="Question",
            placeholder="Pose une question sur le pétrole brut...",
        )
        submit_button = gr.Button("Interroger le RAG")
        answer_output = gr.Textbox(
            label="Réponse générée",
            lines=5,
        )
        sources_output = gr.Textbox(
            label="Contexte récupéré",
            lines=12,
        )

        submit_button.click(
            fn=test_question,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )

    with gr.Tab("Évaluation du dataset"):
        evaluate_button = gr.Button("Lancer l'évaluation")
        results_output = gr.Dataframe(
            label="Résultats",
            interactive=False,
        )
        summary_output = gr.Textbox(label="Résumé")

        evaluate_button.click(
            fn=evaluate_dataset,
            inputs=[],
            outputs=[results_output, summary_output],
        )


if __name__ == "__main__":
    demo.launch()
