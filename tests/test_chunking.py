from gaprag.chunking import chunk_markdown


def test_chunk_preserves_headings_and_source():
    md = "# Title\n\nFirst paragraph.\n\n## Sub\n\nSecond paragraph."
    chunks = chunk_markdown(md, "doc.md", chunk_size=100, chunk_overlap=10)
    assert chunks
    assert all(c.source == "doc.md" for c in chunks)
    assert any(c.heading == "Sub" for c in chunks)


def test_chunk_ids_are_unique():
    md = "# H\n\n" + ("word " * 500)
    chunks = chunk_markdown(md, "d.md", chunk_size=200, chunk_overlap=50)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_splits_into_multiple_windows():
    text = "# H\n\n" + ("sentence about embeddings. " * 200)
    chunks = chunk_markdown(text, "d.md", chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 1


def test_embedding_text_prefixes_heading():
    chunks = chunk_markdown("# Heading\n\nBody text.", "d.md")
    assert chunks[0].embedding_text.startswith("Heading")
