# Chunking Strategies

Chunking is the step that splits documents into smaller passages before embedding.
It is one of the highest-leverage decisions in a RAG system: chunks that are too
large dilute the embedding and waste context; chunks that are too small lose the
surrounding meaning needed to answer a question.

## Chunk size and overlap

- **Chunk size** is usually measured in tokens or characters. A common starting
  range is 200-800 tokens. Smaller chunks give more precise retrieval; larger
  chunks give more context per chunk.
- **Overlap** repeats a small slice of text (for example 10-20% of the chunk) at
  the boundary between consecutive chunks so that a sentence split across a
  boundary is not lost. Overlap trades some storage for higher recall.

## Structure-aware chunking

Splitting on document structure — headings, paragraphs, list items, code blocks —
produces more coherent chunks than a blind fixed-size split. A good strategy is to
split on headings first, then pack paragraphs into chunks up to the target size,
carrying the heading text into each chunk as a prefix so the passage keeps its
context.

## Metadata

Every chunk should carry metadata: the source document, the section heading, and a
position or chunk id. Metadata enables citations, per-tenant filtering, and lets you
trace a retrieved passage back to its origin during debugging and evaluation.

## Practical guidance

- Start with structure-aware chunks of ~500-800 characters and ~15% overlap, then
  tune using retrieval metrics on a labeled question set.
- Keep the exact source text in the chunk so citations quote the document verbatim.
- Re-chunk and re-index when the chunking strategy changes; the index is derived
  data and should be reproducible from the corpus.
