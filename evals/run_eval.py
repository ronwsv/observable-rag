"""Evaluation harness for the RAG pipeline.

Retrieval quality (always, no LLM needed):
  - hit_rate@k : did a relevant chunk make it into the top-k?
  - mrr@k      : how highly ranked was the first relevant chunk?
  - precision@k: what fraction of retrieved chunks were relevant?

Answer quality (optional, set GAPRAG_EVAL_GENERATION=1 and configure an LLM):
  - faithfulness: LLM-as-judge check that the answer is grounded in the context.

The retrieval section is hermetic and cheap, so CI runs it on every push. Results
are printed and written to evals/reports/latest.{json,md}.

Usage:
    python evals/run_eval.py
Config via env, e.g. GAPRAG_EMBEDDING_PROVIDER=hash python evals/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gaprag.config import get_settings  # noqa: E402
from gaprag.ingest import load_corpus  # noqa: E402
from gaprag.providers import build_embedding_provider  # noqa: E402
from gaprag.store import VectorStore  # noqa: E402

DATASET = REPO_ROOT / "evals" / "dataset.jsonl"
REPORTS = REPO_ROOT / "evals" / "reports"


def load_dataset() -> list[dict]:
    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate_retrieval(store: VectorStore, embedder, rows: list[dict], k: int) -> dict:
    hits, rrs, precisions, per_q = [], [], [], []
    query_vecs = embedder.embed([r["question"] for r in rows])
    for row, qvec in zip(rows, query_vecs, strict=True):
        results = store.search(qvec, k=k)
        sources = [chunk.source for chunk, _ in results]
        relevant = row["relevant_source"]
        rank = next((i for i, s in enumerate(sources) if s == relevant), None)
        hit = rank is not None
        rr = 1.0 / (rank + 1) if hit else 0.0
        precision = sources.count(relevant) / max(1, len(sources))
        hits.append(1.0 if hit else 0.0)
        rrs.append(rr)
        precisions.append(precision)
        per_q.append(
            {
                "id": row["id"],
                "hit": hit,
                "first_relevant_rank": (rank + 1) if hit else None,
                "retrieved": sources,
            }
        )
    return {
        "k": k,
        "n": len(rows),
        "hit_rate": round(mean(hits), 4),
        "mrr": round(mean(rrs), 4),
        "precision_at_k": round(mean(precisions), 4),
        "per_question": per_q,
    }


JUDGE_SYSTEM = """You are a strict evaluator. Given a QUESTION, the CONTEXT that was \
retrieved, and an ANSWER, decide if every claim in the ANSWER is supported by the \
CONTEXT. Reply with a single JSON object: {"faithful": true|false, "reason": "..."}."""


def evaluate_generation(rows: list[dict], k: int) -> dict | None:
    """Optional: generate answers and LLM-judge their faithfulness. Needs a real LLM."""
    if os.getenv("GAPRAG_EVAL_GENERATION", "0") != "1":
        return None
    import json as _json
    import re

    from gaprag.providers import build_llm_provider
    from gaprag.rag import RAGPipeline
    from gaprag.retriever import Retriever

    settings = get_settings()
    embedder = build_embedding_provider(
        settings.embedding_provider, settings.embedding_model, region=settings.aws_region
    )
    chunks = load_corpus(
        settings.corpus_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    store = VectorStore(embedder.embed([c.embedding_text for c in chunks]), chunks, embedder.model)
    llm = build_llm_provider(
        settings.llm_provider,
        settings.llm_model,
        ollama_base_url=settings.ollama_base_url,
        region=settings.aws_region,
    )
    pipeline = RAGPipeline(Retriever(store, embedder), llm, top_k=k)

    faithfuls = []
    for row in rows:
        resp = pipeline.answer(row["question"], top_k=k)
        context = "\n".join(c.snippet for c in resp.citations)
        judge = llm.complete(
            JUDGE_SYSTEM,
            f"QUESTION: {row['question']}\n\nCONTEXT:\n{context}\n\nANSWER:\n{resp.answer}",
        )
        match = re.search(r"\{.*\}", judge.text, re.S)
        verdict = _json.loads(match.group(0)) if match else {"faithful": False}
        faithfuls.append(1.0 if verdict.get("faithful") else 0.0)
    return {"n": len(rows), "faithfulness": round(mean(faithfuls), 4)}


def write_reports(report: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    r = report["retrieval"]
    lines = [
        "# Evaluation report",
        "",
        f"- Embedding provider: `{report['embedding_provider']}` "
        f"(model `{report['embedding_model']}`)",
        f"- Index backend: `{report['index_backend']}`",
        f"- Questions: {r['n']}, k = {r['k']}",
        "",
        "## Retrieval",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit rate @{r['k']} | {r['hit_rate']:.3f} |",
        f"| MRR @{r['k']} | {r['mrr']:.3f} |",
        f"| Precision @{r['k']} | {r['precision_at_k']:.3f} |",
    ]
    if report.get("generation"):
        g = report["generation"]
        lines += ["", "## Generation", "", "| Metric | Value |", "|---|---|",
                  f"| Faithfulness | {g['faithfulness']:.3f} |"]
    (REPORTS / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    settings = get_settings()
    rows = load_dataset()
    embedder = build_embedding_provider(
        settings.embedding_provider, settings.embedding_model, region=settings.aws_region
    )
    chunks = load_corpus(
        settings.corpus_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    store = VectorStore(embedder.embed([c.embedding_text for c in chunks]), chunks, embedder.model)

    retrieval = evaluate_retrieval(store, embedder, rows, k=settings.top_k)
    generation = evaluate_generation(rows, k=settings.top_k)

    report = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": embedder.model,
        "index_backend": store.backend,
        "retrieval": retrieval,
        "generation": generation,
    }
    write_reports(report)

    print("=" * 60)
    print(f"Retrieval eval  |  embeddings: {settings.embedding_provider} ({store.backend})")
    print(f"  questions      : {retrieval['n']}  (k={retrieval['k']})")
    print(f"  hit_rate@{retrieval['k']}    : {retrieval['hit_rate']:.3f}")
    print(f"  mrr@{retrieval['k']}         : {retrieval['mrr']:.3f}")
    print(f"  precision@{retrieval['k']}   : {retrieval['precision_at_k']:.3f}")
    if generation:
        print(f"  faithfulness   : {generation['faithfulness']:.3f}")
    print("=" * 60)

    min_hit = float(os.getenv("GAPRAG_EVAL_MIN_HIT_RATE", "0.5"))
    if retrieval["hit_rate"] < min_hit:
        print(f"FAIL: hit_rate {retrieval['hit_rate']:.3f} < threshold {min_hit}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
