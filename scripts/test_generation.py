
from rag_evaluation.generation.answer import generate_answer


def main() -> None:
    
    question = ("Quels éléments chimiques composent le pétrole brut ?"
)
    answer, documents = generate_answer(question, top_k=5)

    print(f"\nQuestion : {question}\n")
    print(f"Réponse :\n{answer}\n")
    print("Sources :")

    for document in documents:
        page = document.metadata.get("page", "?")
        page_number = page + 1 if isinstance(page, int) else page
        print(f"- Page PDF {page_number}")


if __name__ == "__main__":
    main()
