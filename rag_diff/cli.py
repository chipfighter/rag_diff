"""CLI entry point for RAG-Diff.

Provides the `rag-diff` command with subcommands for running test sets,
comparing runs, and managing snapshots.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rag_diff.core.runner import run_testset
from rag_diff.models import RunSnapshot, TestCase
from rag_diff.storage.run_manager import RunManager
from rag_diff.utils.loader import load_adapter

app = typer.Typer(
    name="rag-diff",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Regression testing and diagnostic diffing for RAG pipelines."""


@app.command()
def run(
    testset: str = typer.Option(..., help="Path to the test set JSON file."),
    target: str = typer.Option(..., help="Adapter target in 'module:function' format."),
    output_dir: str = typer.Option(".ragdiff", help="Output directory for run data."),
    concurrency: int = typer.Option(5, help="Max concurrent adapter calls."),
    compare_to: Optional[str] = typer.Option(
        None, help="Run ID to compare against (Phase 2)."
    ),
) -> None:
    """Run the test set against the adapter and record a snapshot."""
    testset_path = Path(testset)
    if not testset_path.exists():
        console.print(f"[red]Error:[/red] Test set not found: {testset}")
        raise typer.Exit(code=1)

    try:
        raw = json.loads(testset_path.read_text(encoding="utf-8"))
        cases = [TestCase(**item) for item in raw]
    except Exception as e:
        console.print(f"[red]Error loading test set:[/red] {e}")
        raise typer.Exit(code=1)

    try:
        adapter = load_adapter(target)
    except Exception as e:
        console.print(f"[red]Error loading adapter:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"\nRunning {len(cases)} case(s) with concurrency={concurrency}...")
    results = asyncio.run(run_testset(adapter, cases, concurrency=concurrency))

    manager = RunManager(base_dir=output_dir)
    run_id = manager.generate_run_id()
    snapshot = RunSnapshot(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        adapter=target,
        testset_path=str(testset_path),
        results=results,
    )
    manager.save(snapshot)

    console.print(f"\n[green]Run complete:[/green] [bold]{run_id}[/bold]")
    console.print(f"  Cases: {len(results)}")
    if results:
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        console.print(f"  Avg latency: {avg_latency:.1f}ms")
    console.print(f"  Snapshot: {output_dir}/runs/{run_id}/")

    if compare_to:
        try:
            baseline = manager.load(compare_to)
            console.print(
                f"\n[bold]Baseline loaded:[/bold] {compare_to} "
                f"({len(baseline.results)} cases)"
            )
        except FileNotFoundError:
            console.print(
                f"[red]Error:[/red] Baseline run '{compare_to}' not found"
            )
            raise typer.Exit(code=1)
