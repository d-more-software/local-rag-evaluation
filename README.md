# RAG Evaluation Project

A Retrieval-Augmented Generation (RAG) application that answers questions about crude oil using a PDF document, hybrid retrieval, and a locally hosted language model. The project also implements an evaluation pipeline to assess answer completeness, faithfulness, and relevance.

## Overview

This project explores how to build and evaluate a RAG system from scratch using Python and open-source tools.

The application retrieves relevant passages from a French technical document about crude oil and uses a local LLM to generate answers grounded in the retrieved context.

A Gradio interface allows users to ask questions, inspect retrieved passages, and evaluate the system against a small reference dataset.

## Features

- **PDF ingestion:** Load and process a technical PDF document.
- **Text chunking:** Split extracted text into smaller passages for retrieval.
- **Semantic search:** Retrieve relevant passages using vector embeddings and FAISS.
- **Hybrid search:** Combine semantic retrieval with BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Local LLM inference:** Generate answers using Ollama and Qwen2.5 1.5B.
- **RAG evaluation:** Assess faithfulness and answer relevance with DeepEval.
- **Completeness evaluation:** Check whether answers contain predefined required facts.
- **Interactive interface:** Test questions and inspect retrieved context using Gradio.

## Architecture

The application follows this pipeline:

1. **Document ingestion:** Extract text from the source PDF.
2. **Chunking:** Split the extracted text into overlapping chunks.
3. **Embedding generation:** Convert chunks into vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Indexing:** Store embeddings in a local FAISS index.
5. **Hybrid retrieval:** Combine semantic search and BM25 keyword search using Reciprocal Rank Fusion.
6. **Answer generation:** Pass the retrieved context and user question to the local Qwen2.5 model through Ollama.
7. **Evaluation:** Compare generated answers with expected answers and required facts.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Dependency management | uv |
| LLM runtime | Ollama |
| Language model | Qwen2.5 1.5B |
| Embeddings | Sentence Transformers |
| Vector store | FAISS |
| Keyword retrieval | BM25 |
| Retrieval strategy | Hybrid search + RRF |
| RAG framework | LangChain |
| Evaluation | DeepEval |
| Interface | Gradio |
| Data analysis | Pandas |
| Code quality | Ruff |
| Testing | Pytest |

## Project Structure

```text
rag-evaluation-projet/
├── apps/
│   ├── app.py
│   └── evaluation_app.py
├── data/
│   ├── documents/
│   ├── evaluation/
│   │   └── golden_dataset.json
│   └── faiss_index/
├── scripts/
│   ├── index_documents.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── evaluate_rag.py
├── src/
│   └── rag_evaluation/
│       ├── config.py
│       ├── generation/
│       ├── ingestion/
│       ├── retrieval/
│       └── evaluation/
├── tests/
│   └── unit/
├── .env
├── pyproject.toml
└── README.md
```

## Requirements

Before running the application, make sure you have:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Git

The local language model and embedding model require additional memory and disk space.

## Installation and Setup

### 1. Clone the repository

```bash
git clone 
cd rag-evaluation-projet
```



### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
RAG_APP_NAME=RAG Evaluation Project
RAG_ENVIRONMENT=development
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_LLM_MODEL=qwen2.5:1.5b
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
RAG_TOP_K=3
RAG_DOCUMENTS_DIR=data/documents
RAG_FAISS_INDEX_DIR=data/faiss_index
```


### 4. Start Ollama and download the model

Make sure Ollama is running, then execute:

```bash
ollama pull qwen2.5:1.5b
```

The embedding model is downloaded automatically when required by the application, subject to network availability.

### 5. Add the source document

Place the PDF document in:

```text
data/documents/
```

The original development dataset uses a French technical document about crude oil. Ensure that you have permission to redistribute the document before publishing it.

### 6. Build the vector index

```bash
uv run python scripts/index_documents.py
```

Run this step again whenever you change the source documents or indexing configuration.

## Usage



### Run the  interface

```bash
uv run python apps/evaluation_app.py
```

The Gradio interface provides two main features:

- **Manual Question Testing:** Submit a question, inspect the generated answer, and review the retrieved context.
- **Dataset Evaluation:** Run the reference questions and view answer completeness, missing facts, and the number of retrieved passages.

### Run the automated evaluation pipeline

```bash
uv run python scripts/evaluate_rag.py
```

This script generates answers for the reference dataset, calculates completeness scores, and runs DeepEval metrics.

## Evaluation Methodology

The project uses a small reference dataset containing five questions about crude oil.

Three evaluation dimensions are considered:

### 1. Answer Completeness

A custom lexical metric checks whether predefined required facts appear in the generated answer.

The score is calculated as:

`Completeness = Number of required facts found / Total required facts`

This metric is simple and interpretable, but it does not understand synonyms or semantic equivalence. A fact can be correct yet receive no credit if its wording differs from the expected expression.

### 2. Faithfulness

DeepEval's `FaithfulnessMetric` evaluates whether the generated answer is supported by the retrieved context.

### 3. Answer Relevancy

DeepEval's `AnswerRelevancyMetric` evaluates how relevant the generated answer is to the original question.

The DeepEval metrics use a locally hosted Ollama model as the LLM judge.

## Initial Results

During local development, the five-question dataset produced the following results:

| Metric | Initial result |
|---|---:|
| Average completeness | 90% |
| Faithfulness average | 0.90 |
| Faithfulness pass rate | 80% |
| Answer relevancy average | 0.83 |
| Answer relevancy pass rate | 60% |
| Overall DeepEval pass rate | 60% |

These are initial development results, not a formal benchmark. The dataset is small, and the LLM judge can misclassify valid answers. Results may also vary depending on the model, retrieved context, and configuration.

## Limitations

- The evaluation dataset contains only five questions and is not representative of all possible queries.
- The local 1.5B-parameter model has limited generation and reasoning capabilities.
- The completeness metric relies on lexical matching rather than semantic understanding.
- LLM-as-a-Judge evaluations are not perfectly reliable and should be interpreted alongside manual inspection.
- Retrieval quality depends on document extraction, chunking, embedding quality, and ranking.
- The current application is designed as a local prototype rather than a production deployment.

## Future Improvements

Potential next steps include:

- Expanding the evaluation dataset with more representative questions.
- Improving completeness evaluation with semantic matching.
- Comparing retrieval configurations and `top_k` values.
- Adding a reranking stage to improve passage ordering.
- Evaluating retrieval quality separately from answer generation.
- Comparing different local language models.
- Adding automated regression tests for the evaluation pipeline.

## Running Tests

Check code quality:

```bash
uv run ruff check .
```

Run the test suite:

```bash
uv run pytest
```

## Key Takeaways

This project demonstrates a practical implementation of a local RAG pipeline, combining semantic and lexical retrieval with LLM-based answer generation.

It also explores an important aspect of AI engineering: evaluating generated answers instead of relying solely on whether the application appears to work.

The project provides a foundation for further experiments with retrieval optimization, evaluation methods, and more advanced RAG architectures.
