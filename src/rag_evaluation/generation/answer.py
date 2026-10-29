
from langchain_core.documents import Document

from rag_evaluation.generation.llm import create_llm
from rag_evaluation.retrieval.hybrid_search import hybrid_search


def generate_answer(
    question: str,
    top_k: int = 3,
) -> tuple[str, list[Document]]:
    """Generate an answer and return its supporting documents."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    documents = hybrid_search(question, top_k=top_k)

    context_parts = []
    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"[Source {index}]\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)
    print("\n--- CONTEXTE RÉCUPÉRÉ ---")
    print(context)
    print("--- FIN DU CONTEXTE ---\n")


    prompt = f"""
Tu es un assistant technique spécialisé dans le pétrole brut.

Réponds en français à la question en utilisant exclusivement le contexte fourni.

Contraintes :
- Réponds en 4 à 6 puces maximum.
- Chaque information ne doit apparaître qu'une seule fois.
- Mentionne tous les constituants explicitement indiqués dans le contexte.
- Associe chaque pourcentage uniquement à l'élément correspondant dans le contexte.
- Distingue les hydrocarbures, les autres éléments et les métaux à l'état de traces.
- N'invente aucune explication, classification ou propriété.
- Si le contexte ne permet pas de préciser un point, 
ne le complète pas avec tes connaissances.
- Ne confonds pas le pétrole brut avec les produits obtenus par raffinage.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""



    response = create_llm().invoke(prompt)
    answer = str(response.content).strip()

    unique_documents = []
    seen_pages = set()

    for document in documents:
        source = document.metadata.get("source", "inconnue")
        page = document.metadata.get("page", "?")
        key = (source, page)

        if key not in seen_pages:
            seen_pages.add(key)
            unique_documents.append(document)

    return answer, unique_documents
