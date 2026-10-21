"""Generate a standalone HTML report from multi-document benchmark results."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/demo/multi_doc_retriever_results.json")
DEFAULT_OUTPUT = Path("reports/multi_doc_benchmark_report.html")
REQUIRED_METHODS = ("tfidf", "semantic")
REQUIRED_METRICS = ("precision_at_k", "recall_at_k", "mrr")


def load_results(path: Path) -> dict[str, Any]:
    """Load and validate the benchmark JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark results not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not isinstance(data.get("methods"), dict):
        raise ValueError("Invalid benchmark JSON: expected a 'methods' object.")

    for method in REQUIRED_METHODS:
        details = data["methods"].get(method)
        if not isinstance(details, dict):
            raise ValueError(f"Missing benchmark method: {method}")

        metrics = details.get("metrics")
        queries = details.get("query_results")

        if not isinstance(metrics, dict) or not isinstance(queries, list):
            raise ValueError(
                f"Method '{method}' must contain metrics and query_results."
            )

        for metric in REQUIRED_METRICS:
            if metric not in metrics:
                raise ValueError(f"Missing metric '{metric}' for '{method}'.")

    return data


def escape(value: Any) -> str:
    """Escape a value before inserting it into HTML."""
    return html.escape(str(value), quote=True)


def format_metric(value: Any) -> str:
    """Format a metric as a percentage or a decimal score."""
    return f"{float(value):.3f}"


def safe_json_for_html(value: Any) -> str:
    """Serialize JSON safely for an embedded script element."""
    serialized = json.dumps(value, ensure_ascii=False)
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_report(data: dict[str, Any]) -> str:
    """Build the complete standalone HTML report."""
    documents = data.get("corpus_documents", [])
    document_count = len(documents)
    chunk_count = int(data.get("corpus_chunks", 0))
    query_count = int(data.get("queries_evaluated", 0))
    k = int(data.get("k", 3))

    tfidf_metrics = data["methods"]["tfidf"]["metrics"]
    semantic_metrics = data["methods"]["semantic"]["metrics"]

    metric_rows = []
    for key, label, explanation in (
        ("precision_at_k", f"Precision@{k}", "Part des résultats retrouvés qui sont pertinents."),
        ("recall_at_k", f"Recall@{k}", "Part des éléments pertinents retrouvés dans le top-k."),
        ("mrr", "MRR", "Rang du premier résultat pertinent, résumé sur les questions."),
    ):
        left = float(tfidf_metrics[key])
        right = float(semantic_metrics[key])
        difference = right - left
        difference_label = f"{difference:+.3f}"

        metric_rows.append(
            f"""
            <tr>
              <th scope="row">
                <span class="metric-name">{escape(label)}</span>
                <span class="metric-description">{escape(explanation)}</span>
              </th>
              <td class="numeric">{left:.3f}</td>
              <td class="numeric">{right:.3f}</td>
              <td class="numeric difference">{difference_label}</td>
            </tr>
            """
        )

    model = data["methods"]["semantic"].get(
        "model", "Modèle d'embeddings non précisé"
    )
    corpus_names = ", ".join(escape(name) for name in documents) or "Non précisé"

    embedded_data = {
        "k": k,
        "methods": {
            method: {
                "description": data["methods"][method].get("description", method),
                "query_results": data["methods"][method]["query_results"],
            }
            for method in REQUIRED_METHODS
        },
    }

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Rapport interactif comparant la recherche TF-IDF et la recherche sémantique sur un corpus de documents PDF.">
  <title>LLM Evaluation Lab — Rapport de benchmark</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #64748b;
      --line: #e2e8f0;
      --surface: #ffffff;
      --background: #f5f7fb;
      --accent: #3659d9;
      --accent-soft: #edf1ff;
      --good: #16794b;
      --good-soft: #e9f8f0;
      --bad: #a83a3a;
      --bad-soft: #fff0f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--background);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .container {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; }}
    header {{
      background: #111b32;
      color: #fff;
      padding: 42px 0 36px;
    }}
    .eyebrow {{
      color: #aebfff;
      text-transform: uppercase;
      letter-spacing: .12em;
      font-size: .76rem;
      font-weight: 700;
    }}
    h1 {{ margin: 10px 0; font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.1; }}
    header p {{ max-width: 760px; color: #d7def0; margin-bottom: 0; }}
    main {{ padding: 28px 0 56px; }}
    section {{ margin-bottom: 30px; }}
    h2 {{ font-size: 1.35rem; margin: 0 0 14px; }}
    .section-intro {{ color: var(--muted); margin: -5px 0 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .card, .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 5px 20px rgba(20, 35, 70, .035);
    }}
    .card-label {{ color: var(--muted); font-size: .88rem; }}
    .card-value {{ font-size: 1.8rem; font-weight: 750; margin-top: 5px; }}
    .card-detail {{ color: var(--muted); font-size: .82rem; overflow-wrap: anywhere; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 540px; }}
    th, td {{ text-align: left; padding: 14px 12px; border-bottom: 1px solid var(--line); }}
    thead th {{ color: var(--muted); font-size: .82rem; font-weight: 650; }}
    tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
    .metric-name, .metric-description {{ display: block; }}
    .metric-description {{ color: var(--muted); font-size: .8rem; font-weight: 400; }}
    .numeric {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .difference {{ font-weight: 700; }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(180px, .45fr);
      gap: 12px;
      margin-bottom: 18px;
    }}
    label {{ display: block; font-size: .86rem; font-weight: 650; margin-bottom: 6px; }}
    input, select {{
      width: 100%;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 11px 12px;
      font: inherit;
      background: white;
      color: var(--ink);
    }}
    input:focus, select:focus, button:focus-visible {{
      outline: 3px solid #b9c7ff;
      outline-offset: 2px;
    }}
    .results-count {{ color: var(--muted); font-size: .88rem; margin-bottom: 12px; }}
    .question {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      margin-bottom: 14px;
      overflow: hidden;
    }}
    .question summary {{
      padding: 16px 18px;
      cursor: pointer;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .question summary:hover {{ background: #f8faff; }}
    .question-body {{ border-top: 1px solid var(--line); padding: 16px; }}
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .method-title {{ font-size: 1rem; margin: 0 0 10px; }}
    .method-subtitle {{ color: var(--muted); font-size: .8rem; margin: -7px 0 12px; }}
    .chunk {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 9px;
      min-width: 0;
    }}
    .chunk.relevant {{ border-color: #9ad5b7; background: var(--good-soft); }}
    .chunk.irrelevant {{ background: #fbfcfe; }}
    .chunk-top {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; }}
    .rank {{ font-weight: 750; }}
    .status {{ font-size: .75rem; font-weight: 750; border-radius: 99px; padding: 3px 8px; }}
    .status.good {{ background: #d0f0df; color: var(--good); }}
    .status.bad {{ background: #edf0f5; color: #475569; }}
    .chunk-name {{ font-size: .84rem; font-weight: 650; overflow-wrap: anywhere; margin-top: 8px; }}
    .chunk-meta {{ font-size: .78rem; color: var(--muted); margin-top: 4px; }}
    .score-track {{ background: #e9edf5; height: 5px; border-radius: 10px; margin-top: 10px; overflow: hidden; }}
    .score-fill {{ height: 100%; background: var(--accent); border-radius: 10px; }}
    .empty {{ padding: 24px; color: var(--muted); text-align: center; }}
    .method-note {{ color: var(--muted); font-size: .88rem; }}
    footer {{ color: var(--muted); font-size: .82rem; padding: 18px 0 0; border-top: 1px solid var(--line); }}
    @media (max-width: 760px) {{
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .columns {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
      header {{ padding: 32px 0; }}
    }}
    @media (max-width: 400px) {{
      .cards {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
      .card {{ padding: 14px; }}
      .card-value {{ font-size: 1.45rem; }}
    }}
  </style>
</head>
<body>
<header>
  <div class="container">
    <div class="eyebrow">LLM Evaluation Lab · Retrieval benchmark</div>
    <h1>Comparer les stratégies de recherche</h1>
    <p>Évaluation reproductible de la recherche lexicale TF-IDF et de la recherche sémantique sur un corpus de PDF, avec des questions annotées et des résultats inspectables.</p>
  </div>
</header>

<main class="container">
  <section aria-labelledby="overview-title">
    <h2 id="overview-title">Vue d'ensemble</h2>
    <div class="cards">
      <article class="card">
        <div class="card-label">Documents PDF</div>
        <div class="card-value">{document_count}</div>
        <div class="card-detail">Corpus multi-document</div>
      </article>
      <article class="card">
        <div class="card-label">Chunks indexés</div>
        <div class="card-value">{chunk_count}</div>
        <div class="card-detail">Segments extraits des PDF</div>
      </article>
      <article class="card">
        <div class="card-label">Questions évaluées</div>
        <div class="card-value">{query_count}</div>
        <div class="card-detail">Jeu de questions annotées</div>
      </article>
      <article class="card">
        <div class="card-label">Top-k</div>
        <div class="card-value">{k}</div>
        <div class="card-detail">Résultats analysés par question</div>
      </article>
    </div>
  </section>

  <section aria-labelledby="metrics-title">
    <h2 id="metrics-title">Comparaison des métriques</h2>
    <p class="section-intro">Les scores sont calculés sur le même jeu de questions. Une valeur MRR plus élevée indique que le premier résultat pertinent apparaît généralement plus haut dans le classement.</p>
    <div class="panel table-wrap">
      <table>
        <thead>
          <tr><th scope="col">Métrique</th><th scope="col" class="numeric">TF-IDF</th><th scope="col" class="numeric">Sémantique</th><th scope="col" class="numeric">Écart (sémantique − TF-IDF)</th></tr>
        </thead>
        <tbody>
          {''.join(metric_rows)}
        </tbody>
      </table>
    </div>
    <p class="method-note">Modèle sémantique : <code>{escape(model)}</code>. Les scores de similarité des deux méthodes ne sont pas directement comparables entre eux ; compare plutôt leur pertinence et leur classement.</p>
  </section>

  <section aria-labelledby="questions-title">
    <h2 id="questions-title">Explorer les résultats question par question</h2>
    <p class="section-intro">Ouvre une question pour comparer les chunks retournés par les deux méthodes. Les résultats pertinents sont mis en évidence.</p>
    <div class="controls">
      <div>
        <label for="search">Rechercher une question ou un document</label>
        <input id="search" type="search" placeholder="Ex. embeddings, Python, rag_guide.pdf">
      </div>
      <div>
        <label for="relevance-filter">Filtrer les questions</label>
        <select id="relevance-filter">
          <option value="all">Toutes les questions</option>
          <option value="both">Les deux méthodes trouvent un résultat pertinent au rang 1</option>
          <option value="semantic">Sémantique meilleure en rang réciproque</option>
          <option value="tfidf">TF-IDF meilleur en rang réciproque</option>
        </select>
      </div>
    </div>
    <div id="results-count" class="results-count" aria-live="polite"></div>
    <div id="questions"></div>
  </section>

  <section aria-labelledby="corpus-title">
    <h2 id="corpus-title">Corpus utilisé</h2>
    <div class="panel">
      <ul>{''.join(f'<li><code>{name}</code></li>' for name in documents)}</ul>
      <p class="method-note">Répertoire source : <code>{escape(data.get("corpus_directory", "Non précisé"))}</code></p>
    </div>
  </section>

  <footer>
    Rapport généré localement par le script Python du projet. Les résultats présentés proviennent du fichier JSON du benchmark ; cette page ne relance pas les modèles.
  </footer>
</main>

<script id="benchmark-data" type="application/json">{safe_json_for_html(embedded_data)}</script>
<script>
(() => {{
  "use strict";
  const data = JSON.parse(document.getElementById("benchmark-data").textContent);
  const searchInput = document.getElementById("search");
  const filterInput = document.getElementById("relevance-filter");
  const questionsElement = document.getElementById("questions");
  const countElement = document.getElementById("results-count");
  const methods = ["tfidf", "semantic"];

  function escapeHtml(value) {{
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }}[char]));
  }}

  function reciprocalRank(result) {{
    const relevant = result.retrieved_chunks.find(chunk => chunk.is_relevant);
    return relevant ? 1 / relevant.rank : 0;
  }}

  function getEntries() {{
    const first = data.methods.tfidf.query_results;
    const semantic = data.methods.semantic.query_results;
    const semanticByQuery = new Map(semantic.map(item => [item.query, item]));
    return first.map(item => ({{
      query: item.query,
      tfidf: item,
      semantic: semanticByQuery.get(item.query)
    }})).filter(item => item.semantic);
  }}

  function renderChunk(chunk) {{
    const relevant = Boolean(chunk.is_relevant);
    const score = Number(chunk.score) || 0;
    const barWidth = Math.max(0, Math.min(100, score * 100));
    return `
      <article class="chunk ${{relevant ? "relevant" : "irrelevant"}}">
        <div class="chunk-top">
          <span class="rank">Rang #${{escapeHtml(chunk.rank)}}</span>
          <span class="status ${{relevant ? "good" : "bad"}}">${{relevant ? "Pertinent" : "Non pertinent"}}</span>
        </div>
        <div class="chunk-name">${{escapeHtml(chunk.document_name)}}</div>
        <div class="chunk-meta">Page ${{escapeHtml(chunk.page_number)}} · Score : ${{score.toFixed(4)}}</div>
        <div class="chunk-meta">${{escapeHtml(chunk.chunk_id)}}</div>
        <div class="score-track" aria-label="Score relatif affiché à titre visuel">
          <div class="score-fill" style="width: ${{barWidth}}%"></div>
        </div>
      </article>`;
  }}

  function renderMethod(method, result) {{
    const label = method === "tfidf" ? "TF-IDF" : "Recherche sémantique";
    const chunks = result.retrieved_chunks || [];
    return `
      <div>
        <h4 class="method-title">${{label}}</h4>
        <p class="method-subtitle">${{escapeHtml(chunks.length)}} résultat(s) retourné(s)</p>
        ${{chunks.length ? chunks.map(renderChunk).join("") : '<p class="method-note">Aucun résultat retourné.</p>'}}
      </div>`;
  }}

  function matchesFilter(entry, filter) {{
    const left = reciprocalRank(entry.tfidf);
    const right = reciprocalRank(entry.semantic);
    if (filter === "both") {{
      return entry.tfidf.retrieved_chunks.some(c => c.rank === 1 && c.is_relevant)
        && entry.semantic.retrieved_chunks.some(c => c.rank === 1 && c.is_relevant);
    }}
    if (filter === "semantic") return right > left;
    if (filter === "tfidf") return left > right;
    return true;
  }}

  function render() {{
    const query = searchInput.value.trim().toLowerCase();
    const filter = filterInput.value;
    const entries = getEntries().filter(entry => {{
      const searchable = [
        entry.query,
        ...entry.tfidf.retrieved_chunks.map(chunk => chunk.document_name),
        ...entry.semantic.retrieved_chunks.map(chunk => chunk.document_name)
      ].join(" ").toLowerCase();
      return searchable.includes(query) && matchesFilter(entry, filter);
    }});

    countElement.textContent = `${{entries.length}} question(s) affichée(s) sur ${{getEntries().length}}.`;
    if (!entries.length) {{
      questionsElement.innerHTML = '<div class="panel empty">Aucune question ne correspond à ces filtres.</div>';
      return;
    }}

    questionsElement.innerHTML = entries.map((entry, index) => `
      <details class="question" ${{index === 0 ? "open" : ""}}>
        <summary>${{escapeHtml(entry.query)}}</summary>
        <div class="question-body">
          <div class="columns">
            ${{renderMethod("tfidf", entry.tfidf)}}
            ${{renderMethod("semantic", entry.semantic)}}
          </div>
        </div>
      </details>`).join("");
  }}

  searchInput.addEventListener("input", render);
  filterInput.addEventListener("change", render);
  render();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML report for a retrieval benchmark."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Benchmark JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    data = load_results(args.input)
    report = build_report(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"Rapport HTML généré : {args.output}")
    print(f"Documents : {len(data.get('corpus_documents', []))}")
    print(f"Chunks : {data.get('corpus_chunks', 0)}")
    print(f"Questions : {data.get('queries_evaluated', 0)}")
    print(f"Fichier : {args.output.resolve()}")


if __name__ == "__main__":
    main()
