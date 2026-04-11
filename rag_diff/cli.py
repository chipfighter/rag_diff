"""CLI entry point for RAG-Diff.

Provides the `rag-diff` command with subcommands for running test sets,
comparing runs, managing snapshots, and bootstrapping new projects.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rag_diff.core.diff import diff_runs
from rag_diff.core.judge import judge_case
from rag_diff.core.runner import run_testset
from rag_diff.models import JudgeVerdict, RunSnapshot, TestCase
from rag_diff.storage.judge_cache import JudgeCache
from rag_diff.storage.run_manager import RunManager
from rag_diff.utils.console import render_diff_report, render_run_summary
from rag_diff.utils.loader import load_adapter

app = typer.Typer(
    name="rag-diff",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Regression testing and diagnostic diffing for RAG pipelines."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    testset: str = typer.Option(..., help="Path to the test set JSON file."),
    target: str = typer.Option(..., help="Adapter target in 'module:function' format."),
    output_dir: str = typer.Option(".ragdiff", help="Output directory for run data."),
    concurrency: int = typer.Option(5, help="Max concurrent adapter calls."),
    compare_to: Optional[str] = typer.Option(
        None, help="Run ID to compare against. Defaults to HEAD (latest run).",
    ),
    judge: bool = typer.Option(
        True, help="Enable LLM-as-Judge evaluation (requires OPENAI_API_KEY).",
    ),
    judge_model: str = typer.Option(
        "gpt-4o-mini", help="Model to use for LLM-as-Judge.",
    ),
    regressions_only: bool = typer.Option(
        False, "--regressions-only", help="Only display regressed cases in output.",
    ),
    dump: bool = typer.Option(
        False, "--dump", help="Export diff report as JSON to .ragdiff/dumps/.",
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

    # --- Run adapter ---
    console.print(f"\nRunning {len(cases)} case(s) with concurrency={concurrency}...")
    results = asyncio.run(run_testset(adapter, cases, concurrency=concurrency))

    # --- Judge (optional) ---
    llm_caller = None
    judge_enabled = False
    if judge:
        llm_caller = _try_create_llm_caller(judge_model)
        judge_enabled = llm_caller is not None

    if judge_enabled:
        console.print(f"Judging {len(results)} case(s) with {judge_model}...")
        cache = JudgeCache(db_path=str(Path(output_dir) / "cache" / "judge_cache_v1.db"))
        results = asyncio.run(_judge_all(results, cache, llm_caller))
        cache_stats = cache.stats()
    else:
        cache_stats = None
        if judge:
            console.print("[yellow]Judge skipped:[/yellow] openai package not installed "
                          "or OPENAI_API_KEY not set. Run with --no-judge to silence.")

    # --- Save snapshot ---
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

    # --- Summary ---
    avg_latency = (sum(r.latency_ms for r in results) / len(results)) if results else 0
    render_run_summary(
        run_id=run_id,
        case_count=len(results),
        avg_latency=avg_latency,
        judge_enabled=judge_enabled,
        cache_stats=cache_stats,
        console=console,
    )
    console.print(f"  Snapshot:    {output_dir}/runs/{run_id}/")

    # --- Diff against baseline ---
    baseline_id = compare_to
    if baseline_id is None:
        baseline_id = manager.get_head()
        # HEAD now points to the run we just saved, so the previous HEAD
        # is what we want to compare against. Check if there are >= 2 runs.
        runs = manager.list_runs()
        if len(runs) >= 2:
            # Previous run is the second-to-last
            prev = [r for r in runs if r != run_id]
            baseline_id = prev[-1] if prev else None
        else:
            baseline_id = None

    if baseline_id:
        try:
            baseline = manager.load(baseline_id)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Baseline run '{baseline_id}' not found")
            raise typer.Exit(code=1)

        report = diff_runs(baseline, snapshot)
        render_diff_report(report, regressions_only=regressions_only, console=console)

        if dump:
            _export_dump(output_dir, report)
    else:
        console.print("\n[dim]No previous run to compare against. "
                      "Run again to see diffs.[/dim]")
        if dump:
            console.print("[dim]--dump skipped: no diff to export.[/dim]")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_SAMPLE_ADAPTER = '''\
"""Sample RAG adapter for rag-diff.

Replace this with your actual RAG pipeline. The function must accept
a query string and return a dict with 'answer' and 'contexts' keys.
"""


async def ask(query: str) -> dict:
    """A dummy adapter that echoes the query."""
    return {
        "answer": f"This is a sample answer to: {query}",
        "contexts": [
            {"id": "doc_1", "text": "Sample context document one."},
            {"id": "doc_2", "text": "Sample context document two."},
        ],
    }
'''

_SAMPLE_TESTSET = json.dumps(
    [
        {"query": "What is RAG?"},
        {"query": "How does vector search work?"},
        {
            "query": "What are embeddings?",
            "expected_answer": "Embeddings are dense vector representations of text.",
        },
    ],
    indent=2,
)


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize."),
) -> None:
    """Bootstrap a new rag-diff project with sample files."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    adapter_path = target / "sample_adapter.py"
    testset_path = target / "sample_testset.json"

    created = []
    for path, content in [(adapter_path, _SAMPLE_ADAPTER), (testset_path, _SAMPLE_TESTSET)]:
        if path.exists():
            console.print(f"[yellow]Skipped:[/yellow] {path} already exists")
        else:
            path.write_text(content, encoding="utf-8")
            created.append(path)
            console.print(f"[green]Created:[/green] {path}")

    if created:
        console.print("\n[bold]Get started:[/bold]")
        console.print(f"  rag-diff run --testset {testset_path} --target {adapter_path}:ask")
    else:
        console.print("\n[dim]Nothing to create — files already exist.[/dim]")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command("list")
def list_runs(
    output_dir: str = typer.Option(".ragdiff", help="Output directory for run data."),
) -> None:
    """List all recorded runs."""
    manager = RunManager(base_dir=output_dir)
    runs = manager.list_runs()
    head = manager.get_head()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    for run_id in runs:
        marker = " [green]<- HEAD[/green]" if run_id == head else ""
        console.print(f"  {run_id}{marker}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _try_create_llm_caller(model: str):
    """Try to create an OpenAI-based LLM caller. Returns None on failure."""
    try:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        async def call(prompt: str) -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""

        return call
    except ImportError:
        return None


async def _judge_all(results, cache, llm_caller):
    """Run judge on all case results."""
    judged = []
    for r in results:
        verdicts = await judge_case(
            query=r.query,
            answer=r.answer,
            contexts=r.contexts,
            context_hashes=r.context_hashes,
            expected_answer=r.expected_answer,
            cache=cache,
            llm_caller=llm_caller,
        )
        judged.append(r.model_copy(update={"judge_verdicts": verdicts}))
    return judged


def _export_dump(output_dir: str, report) -> None:
    """Export a diff report as JSON."""
    dumps_dir = Path(output_dir) / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = dumps_dir / f"diff_{ts}.json"
    dump_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[green]Dump exported:[/green] {dump_path}")
