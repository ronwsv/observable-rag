# LLM Observability

Observability is the practice of instrumenting an LLM application so you can see
what happened on every request: what was retrieved, what prompt was sent, what the
model returned, how long it took, and how much it cost. Without it, an LLM system is
a black box that is impossible to debug or improve in production.

## What to trace per request

- **Inputs:** the user query and any filters.
- **Retrieval:** which chunks were returned and their similarity scores.
- **Prompt:** the exact prompt sent to the model (or a hash of it).
- **Output:** the model's answer and the citations.
- **Latency:** wall-clock time for retrieval and generation, separately.
- **Tokens and cost:** prompt tokens, completion tokens, and the estimated dollar
  cost derived from the model's price per token.

## Structured logging

Emit one structured (JSON) log line per request with a unique request id. Structured
logs can be searched, aggregated, and turned into dashboards, unlike free-form text
logs. A request id ties together every stage of the pipeline for a single call.

## Tracing tools

Purpose-built tools such as Langfuse, LangSmith, or Phoenix capture nested traces
(spans) for retrieval and generation, store prompt/response pairs, and track cost
and latency over time. They make it possible to inspect a single bad answer and to
watch aggregate quality trends.

## Cost and latency control

Track token usage to catch prompt bloat, cache repeated queries, and set budgets.
Latency budgets should separate retrieval from generation so you know which stage to
optimize. Observability data is also the raw material for building evaluation sets:
real production questions become the next batch of test cases.
