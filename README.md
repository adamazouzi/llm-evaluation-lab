# LLM Evaluation Lab

**A reproducible benchmark platform for evaluating and comparing Retrieval-Augmented Generation (RAG) systems.**

> Project status: In development.

## Overview

LLM Evaluation Lab is a Python-based platform designed to evaluate and compare RAG pipeline configurations through controlled, reproducible experiments.

Building a RAG application is not enough: engineers must also understand whether the system retrieves relevant information, generates answers grounded in its context, and meets latency and cost requirements.

This project aims to provide a practical evaluation workflow based on a shared dataset, measurable metrics, and transparent experiment results.

## Objectives

- Evaluate retrieval quality using ranking and relevance metrics.
- Analyze generated answers against reference answers and retrieved context.
- Measure execution latency.
- Estimate API costs when reliable usage and pricing data are available.
- Compare RAG configurations under consistent conditions.
- Store experiment parameters and results for later analysis.
- Visualize performance trade-offs through an interactive dashboard.

## Planned Features

- PDF ingestion, text extraction, and chunking.
- Embedding generation and vector search.
- Configurable retrieval parameters, including chunk size and top-k.
- A documented evaluation dataset with reference answers and relevant passages.
- Retrieval metrics such as Precision@k, Recall@k, and MRR when applicable.
- Answer evaluation with explicit limitations.
- Experiment history and result persistence.
- An interactive dashboard for comparing configurations.
- Automated tests and Docker-based execution.

Features will be marked as implemented only after they have been developed and tested.

## Planned Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Core application and evaluation logic |
| FastAPI | Backend REST API |
| Streamlit | Interactive frontend |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector search |
| SQLite | Experiment and metadata persistence |
| Pandas | Data analysis |
| Plotly | Interactive charts |
| Pytest | Automated testing |
| Docker | Reproducible execution |

The final dependencies and versions will be documented as the implementation progresses.

## Architecture

The planned application consists of five main components:

1. **Document pipeline:** extracts text, creates chunks, and generates embeddings.
2. **RAG pipeline:** retrieves relevant passages and generates context-grounded answers.
3. **Evaluation engine:** computes metrics and records execution measurements.
4. **Backend API:** exposes document, experiment, and result operations.
5. **Frontend and persistence:** displays comparisons and stores experiment history.

## Evaluation Principles

- Use the same evaluation questions across compared configurations.
- Keep reference answers and relevant-passage annotations documented.
- Distinguish retrieval metrics from answer-quality metrics.
- Record configuration parameters and execution errors.
- Report measured results rather than fabricated benchmark scores.
- Document the limitations of automated evaluation.

## Getting Started

Installation and execution instructions will be added once the initial application and its dependencies are implemented.

## Testing

Automated tests will validate core metrics, retrieval behavior, API operations, and experiment persistence.

## Limitations

This project is under development. Benchmark results, supported configurations, and deployment instructions will be documented after they have been implemented and verified.

## Roadmap

- [ ] Initialize the development environment and repository.
- [ ] Implement document ingestion and chunking.
- [ ] Build the retrieval and generation pipeline.
- [ ] Create the evaluation dataset and metrics.
- [ ] Implement experiment execution and persistence.
- [ ] Develop the interactive dashboard.
- [ ] Add automated tests and Docker support.
- [ ] Publish reproducible benchmark results.

## License

License to be selected before the first public release.
