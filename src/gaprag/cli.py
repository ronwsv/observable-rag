"""Command-line interface: ingest, ask, serve, eval."""

from __future__ import annotations

import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import REPO_ROOT, get_settings
from .ingest import ingest as run_ingest
from .observability import configure_logging

app = typer.Typer(add_completion=False, help="Observable RAG — evaluated, provider-agnostic RAG.")
console = Console()


@app.command()
def ingest() -> None:
    """Build the vector index from the corpus."""
    configure_logging(json_logs=False)
    stats = run_ingest()
    table = Table(title="Ingestion complete")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("documents", str(stats.documents))
    table.add_row("chunks", str(stats.chunks))
    table.add_row("embedding model", stats.embedding_model)
    table.add_row("vector dim", str(stats.dim))
    table.add_row("index backend", stats.backend)
    console.print(table)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question."),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of chunks to retrieve."),
) -> None:
    """Ask a question against the indexed corpus."""
    from .rag import build_pipeline  # local import: avoids loading models for other commands

    configure_logging(json_logs=get_settings().json_logs)
    try:
        pipeline = build_pipeline()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    result = pipeline.answer(query, top_k=top_k)
    console.print(
        Panel(result.answer, title=f"Answer (route: {result.route})", border_style="green")
    )

    if result.citations:
        table = Table(title="Sources")
        table.add_column("id")
        table.add_column("heading")
        table.add_column("score", justify="right")
        for c in result.citations:
            table.add_row(c.chunk_id, c.heading, f"{c.score:.3f}")
        console.print(table)

    t = result.trace
    if t is not None:
        console.print(
            f"[dim]retrieval {t.retrieval_ms:.0f}ms · generation {t.generation_ms:.0f}ms · "
            f"{t.total_tokens} tokens · ~${t.cost_usd:.4f} · {t.model}[/dim]"
        )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload for development."),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("gaprag.api:app", host=host, port=port, reload=reload)


@app.command()
def eval() -> None:
    """Run the evaluation harness (evals/run_eval.py)."""
    script = REPO_ROOT / "evals" / "run_eval.py"
    raise typer.Exit(code=subprocess.call([sys.executable, str(script)]))


@app.command()
def mcp() -> None:
    """Run the Model Context Protocol server (exposes RAG as MCP tools over stdio)."""
    from .mcp_server import main as run_mcp

    run_mcp()


if __name__ == "__main__":
    app()
