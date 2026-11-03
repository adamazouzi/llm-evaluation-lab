
import json

import pytest

from scripts.generate_benchmark_report import (
    build_report,
    escape,
    load_results,
    safe_json_for_html,
)


@pytest.fixture
def benchmark_data():
    return {
        "corpus_documents": ["python_guide.pdf", "rag_guide.pdf"],
        "corpus_chunks": 12,
        "queries_evaluated": 24,
        "k": 3,
        "corpus_directory": "data/demo/pdfs",
        "methods": {
            "tfidf": {
                "description": "Recherche lexicale",
                "metrics": {
                    "precision_at_k": 0.33,
                    "recall_at_k": 1.0,
                    "mrr": 0.95,
                },
                "query_results": [
                    {
                        "query": "How can Python automate repetitive work?",
                        "retrieved_chunks": [
                            {
                                "rank": 1,
                                "is_relevant": True,
                                "score": 0.85,
                                "document_name": "python_guide.pdf",
                                "page_number": 2,
                                "chunk_id": "python_guide_2_0",
                            }
                        ],
                    }
                ],
            },
            "semantic": {
                "description": "Recherche sémantique",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "metrics": {
                    "precision_at_k": 0.33,
                    "recall_at_k": 1.0,
                    "mrr": 0.98,
                },
                "query_results": [
                    {
                        "query": "How can Python automate repetitive work?",
                        "retrieved_chunks": [
                            {
                                "rank": 1,
                                "is_relevant": True,
                                "score": 0.91,
                                "document_name": "python_guide.pdf",
                                "page_number": 2,
                                "chunk_id": "python_guide_2_0",
                            }
                        ],
                    }
                ],
            },
        },
    }


def test_load_results_reads_valid_json(tmp_path, benchmark_data):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(benchmark_data), encoding="utf-8")

    result = load_results(path)

    assert result["corpus_chunks"] == 12
    assert "tfidf" in result["methods"]
    assert "semantic" in result["methods"]


def test_load_results_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing.json")


def test_load_results_rejects_invalid_structure(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"invalid": True}), encoding="utf-8")

    with pytest.raises(TypeError):
        load_results(path)


def test_load_results_rejects_missing_method(tmp_path, benchmark_data):
    benchmark_data["methods"].pop("semantic")
    path = tmp_path / "results.json"
    path.write_text(json.dumps(benchmark_data), encoding="utf-8")

    with pytest.raises(TypeError):
        load_results(path)


def test_load_results_rejects_missing_metric(tmp_path, benchmark_data):
    del benchmark_data["methods"]["tfidf"]["metrics"]["mrr"]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(benchmark_data), encoding="utf-8")

    with pytest.raises(TypeError):
        load_results(path)


def test_escape_protects_html():
    result = escape('<script>alert("test")</script>')

    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "&quot;" in result


def test_safe_json_for_html_escapes_html_characters():
    result = safe_json_for_html({"query": "</script><script>alert(1)</script>"})

    assert "</script>" not in result
    assert "\\u003c" in result


def test_build_report_contains_dashboard_sections(benchmark_data):
    report = build_report(benchmark_data)

    assert "<!doctype html>" in report
    assert "Retrieval performance" in report
    assert "Query explorer" in report
    assert "Performance comparison" in report
    assert "Corpus documentaire" in report
    assert "Méthodologie" in report


def test_build_report_contains_benchmark_values(benchmark_data):
    report = build_report(benchmark_data)

    assert "python_guide.pdf" in report
    assert "rag_guide.pdf" in report
    assert "sentence-transformers/all-MiniLM-L6-v2" in report
    assert "24" in report
    assert "12" in report


def test_build_report_embeds_query_results(benchmark_data):
    report = build_report(benchmark_data)

    assert "How can Python automate repetitive work?" in report
    assert "python_guide_2_0" in report
    assert "retrieved_chunks" in report


def test_build_report_escapes_document_names(benchmark_data):
    benchmark_data["corpus_documents"] = ["<script>alert(1)</script>.pdf"]

    report = build_report(benchmark_data)

    assert "<script>alert(1)</script>.pdf" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.pdf" in report
