
from rag_evaluation.ingestion.pipeline import index_documents


def main() -> None:
    chunk_count = index_documents()
    print(f"Indexation terminée : {chunk_count} chunks indexés.")


if __name__ == "__main__":
    main()
