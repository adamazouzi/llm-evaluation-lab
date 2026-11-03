"""Generate a standalone HTML dashboard from retrieval benchmark results."""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/demo/multi_doc_retriever_results.json")
DEFAULT_OUTPUT = Path("reports/multi_doc_benchmark_report.html")
REQUIRED_METHODS = ("tfidf", "semantic")
REQUIRED_METRICS = ("precision_at_k", "recall_at_k", "mrr")


def load_results(path: Path) -> dict[str, Any]:
    """Load and validate benchmark results."""
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark results not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not isinstance(data.get("methods"), dict):
        raise TypeError("Invalid benchmark JSON: expected a 'methods' object.")

    for method in REQUIRED_METHODS:
        details = data["methods"].get(method)

        if not isinstance(details, dict):
            raise TypeError(f"Missing benchmark method: {method}")

        if not isinstance(details.get("metrics"), dict) or not isinstance(
            details.get("query_results"), list
        ):
            raise TypeError(
                f"Method '{method}' must contain metrics and query_results."
            )

        for metric in REQUIRED_METRICS:
            if metric not in details["metrics"]:
                raise TypeError(f"Missing metric '{metric}' for '{method}'.")

    return data


def escape(value: Any) -> str:
    """Escape a value before inserting it into HTML."""
    return html.escape(str(value), quote=True)


def safe_json_for_html(value: Any) -> str:
    """Serialize JSON safely for an embedded script element."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_report(data: dict[str, Any]) -> str:
    """Build the complete standalone HTML dashboard."""
    documents = data.get("corpus_documents", [])
    document_count = len(documents)
    chunk_count = int(data.get("corpus_chunks", 0))
    query_count = int(data.get("queries_evaluated", 0))
    k = int(data.get("k", 3))

    tfidf = data["methods"]["tfidf"]
    semantic = data["methods"]["semantic"]

    tfidf_metrics = tfidf["metrics"]
    semantic_metrics = semantic["metrics"]

    model = semantic.get("model") or "Modèle d'embeddings non précisé"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    definitions = (
        (
            "precision_at_k",
            f"Precision@{k}",
            "Proportion des résultats récupérés qui sont pertinents.",
        ),
        (
            "recall_at_k",
            f"Recall@{k}",
            "Proportion des passages pertinents retrouvés.",
        ),
        (
            "mrr",
            "Mean Reciprocal Rank",
            "Classement du premier résultat pertinent.",
        ),
    )

    rows = []
    chart_rows = []

    for key, label, description in definitions:
        left = float(tfidf_metrics[key])
        right = float(semantic_metrics[key])
        delta = right - left

        delta_class = (
            "positive"
            if delta > 0
            else "negative"
            if delta < 0
            else "neutral"
        )

        rows.append(
            f"""<tr>
                <th scope="row">
                    <strong>{escape(label)}</strong>
                    <span class="table-description">{escape(description)}</span>
                </th>
                <td class="number">{left:.3f}</td>
                <td class="number">{right:.3f}</td>
                <td class="number">
                    <span class="delta {delta_class}">{delta:+.3f}</span>
                </td>
            </tr>"""
        )

        chart_rows.append(
            f"""<div class="metric-row">
                <div class="metric-label">{escape(label)}</div>
                <div class="bars">
                    <div class="bar-line">
                        <div class="bar-track">
                            <div class="bar tfidf"
                                style="width:{max(0, min(100, left * 100)):.2f}%">
                            </div>
                        </div>
                        <span>{left:.3f}</span>
                    </div>
                    <div class="bar-line">
                        <div class="bar-track">
                            <div class="bar semantic"
                                style="width:{max(0, min(100, right * 100)):.2f}%">
                            </div>
                        </div>
                        <span>{right:.3f}</span>
                    </div>
                </div>
            </div>"""
        )

    corpus_items = "".join(
        f'<li><span class="pdf">PDF</span><span>{escape(name)}</span></li>'
        for name in documents
    )

    embedded_data = {
        "methods": {
            method: {
                "query_results": data["methods"][method]["query_results"]
            }
            for method in REQUIRED_METHODS
        }
    }

    mrr_delta = (
        float(semantic_metrics["mrr"]) - float(tfidf_metrics["mrr"])
    )

    def format_value(value: Any, suffix: str, decimals: int = 2) -> str:
        if value is None:
            return "Non disponible"
        return f"{float(value):,.{decimals}f} {suffix}".replace(",", " ")

    tfidf_performance = tfidf.get("performance", {})
    semantic_performance = semantic.get("performance", {})

    performance_definitions = (
        ("Temps d'indexation", "indexing_time_seconds", "s", 4, "method"),
        ("Temps total de recherche", "total_search_time_seconds", "s", 4, "performance"),
        ("Latence moyenne", "average_search_time_ms", "ms", 2, "performance"),
        ("Latence médiane", "median_search_time_ms", "ms", 2, "performance"),
        ("Débit", "queries_per_second", "req/s", 2, "performance"),
    )

    performance_rows = []

    for label, key, unit, decimals, location in performance_definitions:
        left_source = tfidf if location == "method" else tfidf_performance
        right_source = semantic if location == "method" else semantic_performance

        left_value = format_value(left_source.get(key), unit, decimals)
        right_value = format_value(right_source.get(key), unit, decimals)

        performance_rows.append(
            f"""<tr>
                <th scope="row"><strong>{escape(label)}</strong></th>
                <td class="number">{escape(left_value)}</td>
                <td class="number">{escape(right_value)}</td>
            </tr>"""
        )

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description"
content="Dashboard de comparaison des méthodes de recherche TF-IDF et sémantique.">
<title>LLM Evaluation Lab — Retrieval Analytics</title>

<style>
:root {{
    --navy: #111827;
    --ink: #182238;
    --muted: #748096;
    --line: #e7ebf2;
    --bg: #f5f7fb;
    --blue: #5575ed;
    --violet: #8b6ce8;
    --green: #16845d;
    --green-bg: #e8f7f0;
    --shadow: 0 8px 28px rgba(23,32,51,.045);
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.5;
}}

.sidebar {{
    position: fixed;
    inset: 0 auto 0 0;
    width: 238px;
    padding: 25px 18px;
    background: var(--navy);
    color: #fff;
    display: flex;
    flex-direction: column;
    z-index: 2;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 0 7px 34px;
}}

.brand-mark {{
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(135deg,#6684ff,#8c6ee8);
    display: grid;
    place-items: center;
    font-weight: 800;
}}

.brand-name {{
    font-size: 13px;
    font-weight: 750;
}}

.brand-sub {{
    display: block;
    color: #9eabc2;
    font-size: 9px;
    letter-spacing: .12em;
    margin-top: 2px;
}}

.nav-label {{
    font-size: 10px;
    color: #8190aa;
    text-transform: uppercase;
    letter-spacing: .12em;
    padding: 0 11px 10px;
    font-weight: 750;
}}

nav {{
    display: grid;
    gap: 5px;
}}

nav a {{
    color: #b7c1d2;
    text-decoration: none;
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 12px;
}}

nav a:first-child,
nav a:hover {{
    background: #293653;
    color: white;
}}

.nav-icon {{
    display: inline-block;
    width: 25px;
    color: #aab8ff;
}}

.side-foot {{
    margin-top: auto;
    padding: 17px 8px 0;
    border-top: 1px solid #2a354b;
    color: #93a0b5;
    font-size: 10px;
}}

.workspace {{
    margin-left: 238px;
    min-height: 100vh;
}}

.topbar {{
    height: 66px;
    padding: 0 36px;
    background: #fff;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.breadcrumb {{
    color: var(--muted);
    font-size: 11px;
}}

.pill {{
    background: var(--green-bg);
    color: var(--green);
    padding: 6px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 750;
}}

main {{
    max-width: 1400px;
    margin: auto;
    padding: 32px 36px 48px;
}}

.heading {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 25px;
}}

.eyebrow {{
    font-size: 10px;
    color: var(--blue);
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
}}

h1 {{
    font-size: 30px;
    letter-spacing: -.045em;
    margin: 5px 0;
}}

.subtitle {{
    font-size: 12px;
    color: var(--muted);
    margin: 0;
    max-width: 680px;
}}

.generated {{
    font-size: 10px;
    color: var(--muted);
    white-space: nowrap;
    padding-top: 8px;
}}

section {{
    margin-top: 28px;
    scroll-margin-top: 20px;
}}

.section-head {{
    display: flex;
    justify-content: space-between;
    align-items: end;
    margin-bottom: 13px;
}}

h2 {{
    font-size: 15px;
    margin: 0;
}}

.section-note {{
    font-size: 10px;
    color: var(--muted);
    margin: 3px 0 0;
}}

.kpis {{
    display: grid;
    grid-template-columns: repeat(4,minmax(0,1fr));
    gap: 13px;
}}

.card,
.panel {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 12px;
    box-shadow: var(--shadow);
}}

.kpi {{
    padding: 17px;
}}

.kpi-label {{
    font-size: 10px;
    color: var(--muted);
    font-weight: 650;
}}

.kpi-value {{
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -.04em;
    margin-top: 8px;
}}

.kpi-foot {{
    font-size: 9px;
    color: var(--muted);
}}

.two-col {{
    display: grid;
    grid-template-columns: 1.35fr .8fr;
    gap: 13px;
}}

.panel {{
    padding: 19px;
}}

.panel-title {{
    font-size: 12px;
    font-weight: 750;
    margin: 0;
}}

.panel-sub {{
    font-size: 10px;
    color: var(--muted);
    margin: 3px 0 17px;
}}

.legend {{
    font-size: 10px;
    color: var(--muted);
    display: flex;
    gap: 13px;
}}

.dot {{
    width: 8px;
    height: 8px;
    border-radius: 3px;
    display: inline-block;
    margin-right: 5px;
}}

.metric-chart {{
    display: grid;
    gap: 16px;
}}

.metric-row {{
    display: grid;
    grid-template-columns: 100px 1fr;
    gap: 12px;
    align-items: center;
}}

.metric-label {{
    font-size: 10px;
    font-weight: 700;
}}

.bars {{
    display: grid;
    gap: 6px;
}}

.bar-line {{
    display: flex;
    gap: 8px;
    align-items: center;
}}

.bar-track {{
    height: 8px;
    flex: 1;
    background: #f0f2f7;
    border-radius: 20px;
    overflow: hidden;
}}

.bar {{
    height: 100%;
    border-radius: 20px;
}}

.tfidf {{
    background: var(--blue);
}}

.semantic {{
    background: var(--violet);
}}

.bar-line span {{
    font-size: 9px;
    color: var(--muted);
    width: 34px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}}

.insight {{
    background: linear-gradient(135deg,#f0f3ff,#f8f5ff);
    border: 1px solid #e5e8ff;
    border-radius: 10px;
    padding: 15px;
}}

.insight-label {{
    font-size: 9px;
    color: #6c6ed0;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
}}

.insight-value {{
    font-size: 25px;
    font-weight: 800;
    color: #333d86;
    margin: 4px 0;
}}

.insight p,
.method-info {{
    font-size: 10px;
    color: #68738c;
    line-height: 1.7;
}}

.table-wrap {{
    overflow-x: auto;
    margin-top: 13px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    min-width: 540px;
}}

th,
td {{
    padding: 12px;
    border-bottom: 1px solid var(--line);
    text-align: left;
}}

thead th {{
    font-size: 9px;
    color: var(--muted);
}}

tbody th {{
    font-size: 10px;
}}

tbody tr:last-child th,
tbody tr:last-child td {{
    border: 0;
}}

.table-description {{
    display: block;
    color: var(--muted);
    font-size: 9px;
    font-weight: 400;
}}

.number {{
    text-align: right;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
}}

.delta {{
    padding: 4px 7px;
    border-radius: 6px;
    font-size: 9px;
    font-weight: 750;
}}

.positive {{
    background: var(--green-bg);
    color: var(--green);
}}

.negative {{
    background: #fff0f0;
    color: #bd5555;
}}

.neutral {{
    background: #f1f3f7;
    color: var(--muted);
}}

.controls {{
    display: grid;
    grid-template-columns: 1fr 230px;
    gap: 12px;
    margin: 15px 0;
}}

label {{
    display: block;
    font-size: 9px;
    font-weight: 750;
    color: #596579;
    margin-bottom: 5px;
}}

input,
select {{
    width: 100%;
    padding: 10px;
    border: 1px solid #dce2ec;
    border-radius: 8px;
    background: white;
    font: inherit;
    font-size: 10px;
    color: var(--ink);
}}

.results-count {{
    font-size: 10px;
    color: var(--muted);
    margin: 12px 0;
}}

.question {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 10px;
    margin-bottom: 9px;
    box-shadow: var(--shadow);
}}

.question summary {{
    padding: 14px 16px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
}}

.question-body {{
    padding: 0 16px 16px;
}}

.columns {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 13px;
}}

.method-title {{
    font-size: 11px;
    margin: 10px 0 3px;
}}

.method-subtitle {{
    font-size: 9px;
    color: var(--muted);
    margin: 0 0 9px;
}}

.chunk {{
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    background: #fbfcfe;
}}

.chunk.relevant {{
    border-color: #a9ddc7;
    background: #f5fcf8;
}}

.chunk-top {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 6px;
}}

.rank {{
    font-size: 9px;
    font-weight: 750;
}}

.status {{
    font-size: 9px;
    font-weight: 750;
}}

.status.good {{
    color: var(--green);
}}

.status.bad {{
    color: #bd6666;
}}

.chunk-name {{
    font-size: 10px;
    font-weight: 700;
    overflow-wrap: anywhere;
}}

.chunk-meta {{
    font-size: 9px;
    color: var(--muted);
    margin-top: 4px;
    overflow-wrap: anywhere;
}}

.score-track {{
    height: 5px;
    background: #e9edf4;
    border-radius: 10px;
    margin-top: 9px;
    overflow: hidden;
}}

.score-fill {{
    height: 100%;
    background: var(--blue);
    border-radius: 10px;
}}

.corpus-list {{
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 10px;
}}

.corpus-list li {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 10px;
}}

.pdf {{
    background: #fff0f0;
    color: #b94d4d;
    border-radius: 5px;
    padding: 4px 6px;
    font-size: 8px;
    font-weight: 800;
}}

.empty {{
    text-align: center;
    color: var(--muted);
    font-size: 11px;
}}

footer {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    border-top: 1px solid var(--line);
    margin-top: 35px;
    padding-top: 17px;
    color: var(--muted);
    font-size: 9px;
}}

@media (max-width: 1000px) {{
    .sidebar {{
        width: 190px;
    }}

    .workspace {{
        margin-left: 190px;
    }}

    main {{
        padding: 25px 22px;
    }}

    .kpis {{
        grid-template-columns: repeat(2,minmax(0,1fr));
    }}

    .two-col {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 650px) {{
    .sidebar {{
        position: static;
        width: 100%;
        padding: 14px;
    }}

    .brand {{
        padding-bottom: 14px;
    }}

    nav {{
        grid-template-columns: repeat(2,1fr);
    }}

    .side-foot {{
        display: none;
    }}

    .workspace {{
        margin-left: 0;
    }}

    .topbar {{
        padding: 0 16px;
    }}

    main {{
        padding: 22px 14px;
    }}

    .heading {{
        display: block;
    }}

    .generated {{
        padding-top: 12px;
    }}

    .kpis {{
        grid-template-columns: repeat(2,minmax(0,1fr));
    }}

    .controls,
    .columns {{
        grid-template-columns: 1fr;
    }}

    .metric-row {{
        grid-template-columns: 85px 1fr;
    }}

    footer {{
        flex-direction: column;
    }}
}}
</style>
</head>

<body>

<aside class="sidebar">
    <div class="brand">
        <div class="brand-mark">L</div>
        <div>
            <span class="brand-name">LLM Evaluation Lab</span>
            <span class="brand-sub">RESEARCH WORKSPACE</span>
        </div>
    </div>

    <div class="nav-label">Workspace</div>

    <nav aria-label="Navigation">
        <a href="#overview">
            <span class="nav-icon">▦</span>Vue d'ensemble
        </a>
        <a href="#comparison">
            <span class="nav-icon">◫</span>Comparaison
        </a>
        <a href="#performance">
            <span class="nav-icon">◷</span>Performance technique
        </a>
        <a href="#explorer">
            <span class="nav-icon">⌕</span>Query explorer
        </a>
        <a href="#corpus">
            <span class="nav-icon">▤</span>Corpus
        </a>
        <a href="#methodology">
            <span class="nav-icon">ⓘ</span>Méthodologie
        </a>
    </nav>

    <div class="side-foot">
        Research workspace<br>
        Retrieval benchmark · Local report
    </div>
</aside>

<div class="workspace">

    <div class="topbar">
        <span class="breadcrumb">
            Workspace&nbsp; / &nbsp;Retrieval benchmark
        </span>
        <span class="pill">● Benchmark chargé</span>
    </div>

    <main>

        <div class="heading">
            <div>
                <div class="eyebrow">LLM Evaluation Lab · Analytics</div>
                <h1>Retrieval performance</h1>
                <p class="subtitle">
                    Analyse comparative de la recherche lexicale TF-IDF
                    et de la recherche sémantique sur un corpus documentaire multi-PDF.
                </p>
            </div>
            <div class="generated">
                Rapport généré le {escape(generated_at)}
            </div>
        </div>

        <section id="overview">
            <div class="section-head">
                <div>
                    <h2>Vue d'ensemble</h2>
                    <p class="section-note">
                        Périmètre du benchmark et volume évalué.
                    </p>
                </div>
            </div>

            <div class="kpis">
                <article class="card kpi">
                    <div class="kpi-label">DOCUMENTS PDF</div>
                    <div class="kpi-value">{document_count}</div>
                    <div class="kpi-foot">Documents dans le corpus</div>
                </article>

                <article class="card kpi">
                    <div class="kpi-label">CHUNKS INDEXÉS</div>
                    <div class="kpi-value">{chunk_count}</div>
                    <div class="kpi-foot">Segments documentaires</div>
                </article>

                <article class="card kpi">
                    <div class="kpi-label">QUESTIONS ÉVALUÉES</div>
                    <div class="kpi-value">{query_count}</div>
                    <div class="kpi-foot">Questions annotées</div>
                </article>

                <article class="card kpi">
                    <div class="kpi-label">TOP-K</div>
                    <div class="kpi-value">{k}</div>
                    <div class="kpi-foot">Résultats par requête</div>
                </article>
            </div>
        </section>

        <section id="comparison">
            <div class="section-head">
                <div>
                    <h2>Performance comparison</h2>
                    <p class="section-note">
                        Les deux méthodes sont évaluées sur le même jeu de questions.
                    </p>
                </div>
            </div>

            <div class="two-col">

                <div class="panel">
                    <div class="section-head">
                        <div>
                            <h3 class="panel-title">Comparaison des métriques</h3>
                            <p class="panel-sub">
                                Scores de 0 à 1 · Plus élevé = meilleur
                            </p>
                        </div>

                        <div class="legend">
                            <span>
                                <i class="dot" style="background:#5575ed"></i>
                                TF-IDF
                            </span>
                            <span>
                                <i class="dot" style="background:#8b6ce8"></i>
                                Sémantique
                            </span>
                        </div>
                    </div>

                    <div class="metric-chart">
                        {''.join(chart_rows)}
                    </div>
                </div>

                <div class="panel">
                    <h3 class="panel-title">Key takeaway</h3>
                    <p class="panel-sub">Lecture des résultats observés</p>

                    <div class="insight">
                        <div class="insight-label">
                            Écart de MRR · Sémantique vs TF-IDF
                        </div>

                        <div class="insight-value">{mrr_delta:+.3f}</div>

                        <p>
                            La recherche sémantique présente un MRR supérieur
                            sur ce benchmark. Les deux méthodes ont une précision
                            et un rappel identiques.
                        </p>
                    </div>

                    <p class="method-info">
                        <strong>Modèle d'embeddings</strong><br>
                        {escape(model)}
                    </p>
                </div>

            </div>

            <div class="panel table-wrap">
                <h3 class="panel-title" style="margin-bottom:8px">
                    Detailed metrics
                </h3>

                <table>
                    <thead>
                        <tr>
                            <th>Métrique</th>
                            <th class="number">TF-IDF</th>
                            <th class="number">Sémantique</th>
                            <th class="number">Écart</th>
                        </tr>
                    </thead>

                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
        </section>

        <section id="performance">
            <div class="section-head">
                <div>
                    <h2>Performance technique</h2>
                    <p class="section-note">
                        Coût d'indexation, latence de recherche et débit des deux retrievers.
                    </p>
                </div>
            </div>

            <div class="panel table-wrap">
                <h3 class="panel-title" style="margin-bottom:8px">
                    Comparaison des performances
                </h3>
                <table>
                    <thead>
                        <tr>
                            <th>Indicateur</th>
                            <th class="number">TF-IDF</th>
                            <th class="number">Sémantique</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(performance_rows)}
                    </tbody>
                </table>
                <p class="method-info">
                    Les temps sont mesurés localement sur le corpus et le jeu de
                    questions du benchmark. Ils dépendent notamment de la machine,
                    du volume documentaire et du modèle d'embeddings.
                </p>
            </div>
        </section>

        <section id="explorer">
            <div class="section-head">
                <div>
                    <h2>Query explorer</h2>
                    <p class="section-note">
                        Inspectez les passages récupérés et comparez leur classement.
                    </p>
                </div>
            </div>

            <div class="controls">
                <div>
                    <label for="search">
                        RECHERCHER UNE QUESTION OU UN DOCUMENT
                    </label>
                    <input
                        id="search"
                        type="search"
                        placeholder="Ex. embeddings, Python, rag_guide.pdf"
                    >
                </div>

                <div>
                    <label for="relevance-filter">
                        FILTRER LES QUESTIONS
                    </label>

                    <select id="relevance-filter">
                        <option value="all">Toutes les questions</option>
                        <option value="both">
                            Les deux méthodes pertinentes au rang 1
                        </option>
                        <option value="semantic">
                            Sémantique meilleure en MRR
                        </option>
                        <option value="tfidf">
                            TF-IDF meilleur en MRR
                        </option>
                    </select>
                </div>
            </div>

            <div id="results-count" class="results-count" aria-live="polite"></div>
            <div id="questions"></div>
        </section>

        <section id="corpus">
            <div class="section-head">
                <div>
                    <h2>Corpus documentaire</h2>
                    <p class="section-note">
                        Documents PDF utilisés dans cette expérience.
                    </p>
                </div>
            </div>

            <div class="panel">
                <ul class="corpus-list">
                    {corpus_items}
                </ul>

                <p class="method-info">
                    Répertoire source :
                    <code>{escape(data.get("corpus_directory", "Non précisé"))}</code>
                </p>
            </div>
        </section>

        <section id="methodology">
            <div class="section-head">
                <div>
                    <h2>Méthodologie</h2>
                    <p class="section-note">
                        Configuration et interprétation du benchmark.
                    </p>
                </div>
            </div>

            <div class="panel">
                <p class="method-info">
                    <strong>TF-IDF</strong> —
                    {escape(tfidf.get("description", "Recherche lexicale"))}
                </p>

                <p class="method-info">
                    <strong>Recherche sémantique</strong> —
                    {escape(semantic.get("description", "Recherche par embeddings"))}
                </p>

                <p class="method-info">
                    <strong>Precision@k</strong> mesure la proportion des résultats
                    récupérés qui sont pertinents.
                    <strong>Recall@k</strong> mesure la proportion des passages
                    pertinents retrouvés.
                    <strong>MRR</strong> mesure le rang réciproque du premier
                    résultat pertinent.
                </p>

                <p class="method-info">
                    Les scores de similarité bruts des deux méthodes ne sont pas
                    directement comparables. L'analyse porte sur les métriques
                    de pertinence et de classement.
                </p>
            </div>
        </section>

        <footer>
            <span>LLM Evaluation Lab · Retrieval Analytics</span>
            <span>Rapport autonome · Données issues du benchmark JSON</span>
        </footer>

    </main>
</div>

<script id="benchmark-data" type="application/json">{safe_json_for_html(embedded_data)}</script>

<script>
(() => {{
    "use strict";

    const data = JSON.parse(
        document.getElementById("benchmark-data").textContent
    );

    const search = document.getElementById("search");
    const filter = document.getElementById("relevance-filter");
    const output = document.getElementById("questions");
    const count = document.getElementById("results-count");

    function esc(value) {{
        return String(value ?? "").replace(/[&<>"']/g, char => ({{
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }}[char]));
    }}

    function rr(result) {{
        const first = (result.retrieved_chunks || []).find(
            chunk => chunk.is_relevant
        );

        return first ? 1 / first.rank : 0;
    }}

    function entries() {{
        const semantic = new Map(
            data.methods.semantic.query_results.map(
                item => [item.query, item]
            )
        );

        return data.methods.tfidf.query_results
            .map(item => ({{
                query: item.query,
                tfidf: item,
                semantic: semantic.get(item.query)
            }}))
            .filter(item => item.semantic);
    }}

    function chunkCard(chunk) {{
        const relevant = Boolean(chunk.is_relevant);
        const score = Number(chunk.score) || 0;
        const width = Math.max(0, Math.min(100, score * 100));

        return `
            <article class="chunk ${{relevant ? "relevant" : ""}}">
                <div class="chunk-top">
                    <span class="rank">Rang #${{esc(chunk.rank)}}</span>
                    <span class="status ${{relevant ? "good" : "bad"}}">
                        ${{relevant ? "Pertinent" : "Non pertinent"}}
                    </span>
                </div>

                <div class="chunk-name">
                    ${{esc(chunk.document_name)}}
                </div>

                <div class="chunk-meta">
                    Page ${{esc(chunk.page_number)}} · Score : ${{score.toFixed(4)}}
                </div>

                <div class="chunk-meta">
                    ${{esc(chunk.chunk_id)}}
                </div>

                <div class="score-track">
                    <div class="score-fill" style="width:${{width}}%"></div>
                </div>
            </article>
        `;
    }}

    function methodColumn(label, result) {{
        const chunks = result.retrieved_chunks || [];

        return `
            <div>
                <h4 class="method-title">${{label}}</h4>
                <p class="method-subtitle">
                    ${{chunks.length}} résultat(s)
                </p>

                ${{chunks.length
                    ? chunks.map(chunkCard).join("")
                    : '<p class="method-info">Aucun résultat.</p>'
                }}
            </div>
        `;
    }}

    function matches(item) {{
        const left = rr(item.tfidf);
        const right = rr(item.semantic);

        if (filter.value === "both") {{
            return (
                item.tfidf.retrieved_chunks.some(
                    c => c.rank === 1 && c.is_relevant
                ) &&
                item.semantic.retrieved_chunks.some(
                    c => c.rank === 1 && c.is_relevant
                )
            );
        }}

        if (filter.value === "semantic") {{
            return right > left;
        }}

        if (filter.value === "tfidf") {{
            return left > right;
        }}

        return true;
    }}

    function render() {{
        const term = search.value.trim().toLowerCase();
        const all = entries();

        const visible = all.filter(item => {{
            const names = [
                ...(item.tfidf.retrieved_chunks || []),
                ...(item.semantic.retrieved_chunks || [])
            ].map(chunk => chunk.document_name);

            return (
                [item.query, ...names]
                    .join(" ")
                    .toLowerCase()
                    .includes(term) &&
                matches(item)
            );
        }});

        count.textContent =
            `${{visible.length}} question(s) affichée(s) sur ${{all.length}}.`;

        if (!visible.length) {{
            output.innerHTML = `
                <div class="panel empty">
                    Aucune question ne correspond à ces filtres.
                </div>
            `;
            return;
        }}

        output.innerHTML = visible.map((item, index) => `
            <details class="question" ${{index === 0 ? "open" : ""}}>
                <summary>${{esc(item.query)}}</summary>

                <div class="question-body">
                    <div class="columns">
                        ${{methodColumn("TF-IDF", item.tfidf)}}
                        ${{methodColumn("Recherche sémantique", item.semantic)}}
                    </div>
                </div>
            </details>
        `).join("");
    }}

    search.addEventListener("input", render);
    filter.addEventListener("change", render);

    render();
}})();
</script>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML dashboard for a retrieval benchmark."
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    data = load_results(args.input)
    report = build_report(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Dashboard HTML généré : {args.output}")
    print(f"Documents : {len(data.get('corpus_documents', []))}")
    print(f"Chunks : {data.get('corpus_chunks', 0)}")
    print(f"Questions : {data.get('queries_evaluated', 0)}")
    print(f"Fichier : {args.output.resolve()}")


if __name__ == "__main__":
    main()