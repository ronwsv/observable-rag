# Evaluating RAG Systems

You cannot improve what you do not measure. A RAG system has two stages, and each
is evaluated differently: **retrieval** (did we fetch the right context?) and
**generation** (did we produce a faithful, relevant answer?).

## Retrieval metrics

These need a labeled set of questions, each tagged with the document (or chunk)
that contains the answer.

- **Hit rate @ k (recall@k):** the fraction of questions for which at least one
  relevant chunk appears in the top-k retrieved results. It answers "did the right
  context make it into the prompt at all?"
- **MRR (Mean Reciprocal Rank):** the average of 1/rank of the first relevant
  chunk. It rewards putting the relevant chunk near the top, not just anywhere in
  the top-k.
- **Precision@k** measures how many of the retrieved chunks are relevant, which
  matters when extra chunks dilute the context.

Because most RAG errors are retrieval errors, retrieval metrics are the fastest way
to find and fix quality problems, and they are cheap because they need no LLM.

## Generation metrics

Judging free-form answers usually uses an **LLM-as-judge**: a separate model scores
the answer against the question and the retrieved context.

- **Faithfulness (groundedness):** is every claim in the answer supported by the
  retrieved context? This is the direct measure of hallucination.
- **Answer relevance:** does the answer actually address the question?
- **Context precision/recall:** frameworks such as RAGAS score how well the
  retrieved context supports the reference answer.

## Building an evaluation set

Start with 20-50 real questions, each with a reference answer and the source
document. Keep the set in version control so results are comparable over time. Run
the evaluation on every change to chunking, the embedding model, top-k, or the
prompt, and treat a drop in hit rate or faithfulness as a regression.

## Offline vs online

Offline evaluation on a fixed set catches regressions before deploy. Online signals
— user thumbs up/down, citation click-through, and escalation rate — validate real
usefulness after deploy. Mature teams use both.
