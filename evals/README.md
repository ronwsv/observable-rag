# Evaluation

RAG is two systems in one, so it is measured in two places.

## Retrieval (always, no LLM required)

Each question in [`dataset.jsonl`](dataset.jsonl) is labeled with the source
document that contains its answer. The harness retrieves the top-k chunks and
computes:

- **hit_rate@k** — did a relevant chunk make it into the top-k?
- **mrr@k** — how highly ranked was the first relevant chunk?
- **precision@k** — what fraction of retrieved chunks were relevant?

Because this needs no LLM, it is cheap and hermetic, and it runs in CI on every
push (with a minimum hit-rate gate so a retrieval regression fails the build).

```bash
python evals/run_eval.py
# or pin the backend:
GAPRAG_EMBEDDING_PROVIDER=hash python evals/run_eval.py
```

Reports are written to `evals/reports/latest.{json,md}`.

## Generation (optional, needs a real LLM)

With `GAPRAG_EVAL_GENERATION=1` and a configured LLM (Ollama / OpenAI / Anthropic),
the harness also generates answers and uses an **LLM-as-judge** to score
**faithfulness** — whether every claim in the answer is supported by the retrieved
context. This is the direct measure of hallucination.

```bash
GAPRAG_EVAL_GENERATION=1 GAPRAG_LLM_PROVIDER=ollama python evals/run_eval.py
```

## Extending the set

Add rows to `dataset.jsonl`. Keep it in version control so results are comparable
over time, and re-run on every change to chunking, the embedding model, top-k, or
the prompt — treat a drop in hit rate or faithfulness as a regression.
