# Prompt Engineering for RAG

The prompt is where retrieval meets generation. A good RAG prompt turns retrieved
passages into a grounded, cite-able answer and refuses to invent facts that are not
in the context.

## Anatomy of a RAG prompt

- **System instruction:** define the assistant's role and the hard rules — answer
  only from the provided context, cite sources, and say "I don't know" when the
  context does not contain the answer.
- **Context block:** the retrieved chunks, each labeled with its source id so the
  model can cite them and so the answer is traceable.
- **Question:** the user's query, kept separate from the context to avoid prompt
  injection from document content.

## Rules that reduce hallucination

1. Explicitly instruct: "Use only the context below. If the answer is not there,
   say you don't know." This single rule is the most effective hallucination
   guardrail in RAG.
2. Ask the model to cite the source id next to each claim.
3. Keep the context ordered by relevance; put the strongest chunk first.

## Prompt injection

Because retrieved documents are untrusted text, they can contain instructions that
try to hijack the model ("ignore previous instructions..."). Mitigations include
clearly separating instructions from data, never executing instructions found in
retrieved content, and treating document text strictly as reference material.

## Temperature and determinism

For factual RAG answers, use a low temperature (near 0) so the output is stable and
grounded. Higher temperature increases variety and is only useful for creative
tasks, not for question answering over documents.
