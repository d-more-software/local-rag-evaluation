
from rag_evaluation.retrieval.search import search_documents


def main() -> None:
    question = "Quelle est la composition chimique du pétrole brut ?"
    documents = search_documents(question, top_k=3)

    print(f"\nQuestion : {question}\n")

    for rank, document in enumerate(documents, start=1):
        page = document.metadata.get("page", "?")
        source = document.metadata.get("source", "source inconnue")

        page_label = int(page) + 1 if isinstance(page, int) else page

        print(
            f"--- Résultat {rank} | Page PDF {page_label} ---"
        )

        print(f"Source : {source}")
        print(document.page_content[:700])
        print()


if __name__ == "__main__":
    main()
