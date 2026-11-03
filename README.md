# LLM Evaluation Lab

**A reproducible benchmark platform for evaluating and comparing Retrieval-Augmented Generation (RAG) retrieval systems.**

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Tests](https://img.shields.io/badge/Tests-70%20passed-success)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

## Overview

LLM Evaluation Lab is a Python-based platform designed to evaluate and compare RAG retrieval systems through reproducible experiments.

Retrieval quality is a critical component of a RAG system. Even a powerful language model cannot provide reliable answers if the retrieved context is incomplete or irrelevant.

This project provides an evaluation workflow to measure retrieval effectiveness, compare retrieval strategies, and analyze performance trade-offs using documented datasets and quantitative metrics.

The platform currently supports keyword-based and semantic retrieval, a RAG pipeline, Azure OpenAI integration, a REST API, and HTML benchmark reports.

## Key Features

- **PDF ingestion:** Extract text from PDF documents using PyMuPDF.
- **Document chunking:** Split extracted content into configurable text chunks with metadata.
- **Multiple retrieval strategies:**
  - TF-IDF keyword-based retrieval.
  - Semantic retrieval using Sentence Transformers.
- **Retrieval evaluation:** Compute Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- **Performance benchmarking:** Measure indexing time, search latency, and query throughput.
- **RAG pipeline:** Retrieve relevant context and generate answers using configurable LLM providers.
- **LLM integration:** Demo provider and Azure OpenAI integration.
- **REST API:** Expose RAG functionality through FastAPI.
- **HTML dashboard:** Generate standalone reports to visualize and compare benchmark results.
- **Automated testing:** Validate core components with Pytest.
- **Code quality:** Apply static analysis and linting with Ruff.

## Architecture

The platform is organized into five main stages:

```text
                 PDF Documents
                       |
                       v
              Document Ingestion
              (PyMuPDF Extraction)
                       |
                       v
                 Text Chunking
                       |
                       v
              Retrieval Indexing
                 /         \
                v           v
             TF-IDF      Semantic
             Search      Search
                \           /
                 \         /
                  v       v
              Evaluation Engine
             (Precision, Recall,
                    MRR)
                       |
                       v
             Performance Analysis
           (Latency, Indexing, QPS)
                       |
                       v
                HTML Dashboard
```

The RAG application also exposes a separate answer-generation workflow:

```text
User Question
      |
      v
Retriever
      |
      v
Relevant Context
      |
      v
LLM Provider
(Demo / Azure OpenAI)
      |
      v
Generated Answer + Sources
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12+ | Core application and evaluation logic |
| FastAPI | REST API |
| Uvicorn | ASGI server |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Semantic embeddings |
| Scikit-learn | TF-IDF retrieval |
| NumPy | Numerical operations |
| OpenAI SDK | LLM integration |
| Python-dotenv | Environment configuration |
| Pytest | Automated testing |
| HTTPX | API testing |
| Ruff | Code quality and linting |

## Benchmark Results

An extended retrieval benchmark was conducted using three technical documents:

- Machine Learning with Limited Data
- Python Tutorial
- Apache Spark Quick Start

### Benchmark Dataset

| Indicator | Value |
|---|---:|
| PDF documents | 3 |
| Document chunks | 510 |
| Evaluation questions | 36 |
| Relevant question–chunk associations | 59 |
| Unique annotated chunks | 50 |
| Retrieval cutoff | K = 3 |

### TF-IDF vs Semantic Retrieval

| Metric | TF-IDF | Semantic |
|---|---:|---:|
| Precision@3 | 0.361 | **0.370** |
| Recall@3 | **0.722** | 0.692 |
| MRR | **0.681** | 0.625 |
| Indexing time (s) | **0.0229** | 4.5139 |
| Average search time (ms) | **0.368** | 6.844 |
| Queries per second | **2718.9** | 146.1 |

### Observations

- Semantic retrieval achieved slightly higher Precision@3.
- TF-IDF achieved higher Recall@3 and MRR on this evaluation dataset.
- TF-IDF required substantially less indexing time.
- TF-IDF also achieved lower average search latency and higher query throughput.

These results illustrate that semantic retrieval is not necessarily superior to keyword-based retrieval for every dataset.

**Important:** These measurements come from a local experiment involving three documents and 36 questions. They should not be interpreted as universal performance conclusions.

### Benchmark Report

The standalone HTML dashboard is available at:

`reports/extended_multi_doc_benchmark_report.html`

## Getting Started

### Requirements

- Python 3.12 or later
- pip
- Git

### Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd llm-evaluation-lab
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

### Configuration

Create a local environment file:

```bash
cp .env.example .env
```

The default configuration uses the demo provider and demo corpus:

```env
LLM_PROVIDER=demo
RAG_CORPUS=demo
RETRIEVER_TYPE=tfidf
```

Available configuration options:

| Variable | Supported values | Default |
|---|---|---|
| `LLM_PROVIDER` | `demo`, `azure` | `demo` |
| `RAG_CORPUS` | `demo`, `pdf` | `demo` |
| `RETRIEVER_TYPE` | `tfidf`, `semantic` | `tfidf` |

For Azure OpenAI integration, configure the following variables in your local `.env` file:

```env
LLM_PROVIDER=azure

AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-10-21
```

Never commit your `.env` file or expose API credentials.

**Note:** The API's PDF mode currently loads the demonstration corpus from `data/demo/multi_doc_corpus`. The extended three-document corpus is used by the standalone benchmark scripts.

## Running the API

Start the FastAPI application:

```bash
uvicorn backend.app.main:app --reload
```

The API is available at:

`http://127.0.0.1:8000`

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/ask` | Submit a question to the RAG pipeline |

### Interactive API Documentation

FastAPI automatically provides interactive API documentation through Swagger UI:

`http://127.0.0.1:8000/docs`

### Test the Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Submit a Question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Retrieval-Augmented Generation (RAG)?"}'
```

The response contains the generated answer, retrieved sources, and request metadata.

## Running the Tests

Run the complete test suite:

```bash
pytest -v
```

Run static code checks:

```bash
ruff check .
```

## Running the Extended Benchmark

The extended benchmark compares TF-IDF and semantic retrieval using the same evaluation questions and annotations.

```bash
python scripts/compare_multi_doc_retrievers.py \
  --corpus-dir data/extended/multi_doc_corpus \
  --annotations data/extended/multi_doc_questions.json \
  --output data/extended/multi_doc_retriever_results.json \
  -k 3
```

### Generate the HTML Dashboard

```bash
python scripts/generate_benchmark_report.py \
  --input data/extended/multi_doc_retriever_results.json \
  --output reports/extended_multi_doc_benchmark_report.html
```

Open the generated HTML file in a browser to inspect the benchmark results.

## Evaluation Methodology

The benchmark uses manually annotated questions and relevant chunk identifiers.

For each question, the retriever returns its top-K results, which are compared against the reference annotations.

The evaluation engine computes:

- **Precision@K:** Proportion of retrieved chunks that are relevant.
- **Recall@K:** Proportion of relevant chunks retrieved in the top-K results.
- **MRR:** Ranking quality based on the reciprocal rank of the first relevant result.

Performance measurements include:

- Indexing time.
- Total search time.
- Average and median search latency.
- Queries per second.

Using the same evaluation questions and annotations for both retrieval strategies makes the comparison more consistent.

## Project Structure

```text
llm-evaluation-lab/
├── backend/
│   └── app/
│       ├── main.py
│       └── services/
├── evaluation/
├── data/
│   ├── demo/
│   └── extended/
│       ├── multi_doc_corpus/
│       ├── sources/
│       ├── multi_doc_questions.json
│       ├── machine_learning_questions.json
│       ├── spark_questions.json
│       ├── multi_doc_retriever_results.json
│       └── ATTRIBUTIONS.md
├── reports/
│   ├── multi_doc_benchmark_report.html
│   └── extended_multi_doc_benchmark_report.html
├── scripts/
│   ├── compare_multi_doc_retrievers.py
│   ├── generate_benchmark_report.py
│   └── extract_html_to_pdf.py
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

## Testing and Code Quality

The project currently includes automated tests covering:

- PDF extraction and chunking.
- Keyword and semantic retrieval.
- Retrieval evaluation metrics.
- Multi-document benchmarking.
- HTML report generation.
- RAG pipeline behavior.
- FastAPI endpoints.
- LLM provider configuration.
- Azure OpenAI integration.

Latest validation:

- **70 automated tests passed.**
- **Ruff: all checks passed.**

## Current Limitations

- The extended benchmark contains only three documents and 36 evaluation questions.
- Results depend on the selected documents, annotations, embedding model, and retrieval configuration.
- Semantic retrieval has a higher indexing and search cost in the current local experiment.
- Benchmark measurements are local observations, not production-scale performance guarantees.
- Answer-quality evaluation and comprehensive cost benchmarking remain areas for further development.

## Roadmap

- [x] Implement PDF extraction and document chunking.
- [x] Implement TF-IDF retrieval.
- [x] Implement semantic retrieval with Sentence Transformers.
- [x] Implement retrieval evaluation metrics.
- [x] Build a multi-document benchmark.
- [x] Generate HTML benchmark reports.
- [x] Add automated tests and code-quality checks.
- [x] Add a FastAPI application.
- [x] Integrate Azure OpenAI.
- [ ] Improve benchmark dataset diversity and annotation coverage.
- [ ] Expand answer-generation evaluation.
- [ ] Improve experiment reproducibility and configuration tracking.
- [ ] Document deployment and API usage further.
- [ ] Add continuous integration with GitHub Actions.

## Dataset Sources and Attribution

The extended benchmark uses third-party technical documents.

Their original copyright and license notices are documented in:

`data/extended/ATTRIBUTIONS.md`

The documents do not necessarily share the same license. Check the original source and applicable terms before redistributing corpus files.

## License

A license for the project code will be selected before public distribution. Dataset and third-party document licenses remain separate from the project code license.
