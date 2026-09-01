"""Structure-aware markdown chunking.

Documents are split on headings first, then packed into overlapping character
windows. Each chunk carries its source file and section heading so the heading
context travels into the embedding and citations can quote the origin.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """A retrievable passage plus the metadata needed for citation and debugging."""

    id: str
    source: str
    heading: str
    text: str

    @property
    def embedding_text(self) -> str:
        """Text actually embedded: heading prefixed so the passage keeps its context."""
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(id=d["id"], source=d["source"], heading=d["heading"], text=d["text"])


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into character windows of ``size`` with ``overlap`` between them."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(text), step):
        window = text[start : start + size].strip()
        if window:
            windows.append(window)
        if start + size >= len(text):
            break
    return windows


def chunk_markdown(
    text: str, source: str, *, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    """Chunk one markdown document into a list of :class:`Chunk`."""
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, current_body))

    chunks: list[Chunk] = []
    index = 0
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        for window in _windows(body, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(id=f"{source}#{index}", source=source, heading=heading, text=window)
            )
            index += 1
    return chunks
