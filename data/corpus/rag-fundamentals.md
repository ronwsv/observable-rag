# RAG Fundamentals

Retrieval-Augmented Generation (RAG) is a pattern that grounds a large language
model (LLM) in an external knowledge source at inference time. Instead of relying
only on the parameters learned during pre-training, a RAG system retrieves
relevant passages from a corpus and passes them to the model as context, so the
answer is conditioned on up-to-date, domain-specific, and verifiable information.

## Why use RAG

- **Freshness:** the knowledge base can change without retraining the model.
- **Grounding:** answers can cite sources, which reduces hallucination.
- **Cost:** retrieval is far cheaper than fine-tuning for most knowledge tasks.
- **Access control:** documents can be filtered per user or tenant before retrieval.

Use RAG when answers must reflect a specific, changing body of documents
(support articles, policies, product manuals, internal wikis). Prefer fine-tuning
when you need to change the model's *behavior or style* rather than give it new facts.

## The RAG pipeline

1. **Ingestion (offline):** load documents, split them into chunks, compute an
   embedding for each chunk, and store the vectors in an index.
2. **Retrieval (online):** embed the user query, search the index for the nearest
   chunks, and return the top-k passages with their similarity scores.
3. **Generation (online):** build a prompt from the retrieved passages plus the
   question, call the LLM, and return the answer together with citations.

## Grounding and citations

A well-behaved RAG system instructs the model to answer **only** from the provided
context and to say "I don't know" when the context is insufficient. Returning the
source of each retrieved chunk lets users verify the answer and builds trust. This
"answer only from context" instruction is the single most effective guardrail
against hallucination in a RAG application.

## Common failure modes

- **Retrieval miss:** the relevant chunk is never retrieved, so the model cannot
  answer correctly no matter how good it is. Most RAG quality problems are
  retrieval problems, not generation problems.
- **Context dilution:** too many low-relevance chunks bury the useful one.
- **Lost in the middle:** models attend less to the middle of a long context, so
  ordering retrieved chunks by relevance matters.
