from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF and return its pages as LangChain documents."""
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")

    documents = PyMuPDFLoader(str(file_path)).load()

    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError(f"No readable text found in PDF: {file_path}")

    return documents
