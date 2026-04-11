"""Rich terminal rendering for diff reports and run summaries."""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_diff.core.diff import CaseDiff, DiffReport, TriageLabel

# Default console instance used by CLI
_default_console = Console()

_TRIAGE_ICONS = {
    TriageLabel.RETRIEVAL: "[yellow]Suspected Retrieval Issue[/yellow]",
    TriageLabel.MODEL: "[magenta]Suspected Model Issue[/magenta]",
    TriageLabel.UNKNOWN: "[dim]Unknown (no judge data)[/dim]",
}

_STATUS_ICONS = {
    "regression": "[red]REGRESSION[/red]",
    "improvement": "[green]IMPROVEMENT[/green]",
    "changed": "[yellow]CHANGED[/yellow]",
    "unchanged": "[dim]UNCHANGED[/dim]",
}


def render_run_summary(
    run_id: str,
    case_count: int,
    avg_latency: float,
    judge_enabled: bool = False,
    cache_stats: Optional[dict] = None,
    console: Optional[Console] = None,
) -> None:
    """Render a summary after a run completes."""
    con = console or _default_console
    con.print()
    con.print(f"[green]Run complete:[/green] [bold]{run_id}[/bold]")
    con.print(f"  Cases:       {case_count}")
    con.print(f"  Avg latency: {avg_latency:.1f}ms")
    if judge_enabled:
        con.print(f"  Judge:       enabled")
        if cache_stats:
            con.print(f"  Cache:       {cache_stats.get('total', 0)} entries")


def render_diff_report(
    report: DiffReport,
    regressions_only: bool = False,
    console: Optional[Console] = None,
) -> None:
    """Render a full diff report to the terminal."""
    con = console or _default_console

    # --- Summary bar ---
    summary_parts = []
    if report.regressions:
        summary_parts.append(f"[red]{report.regressions} regression(s)[/red]")
    else:
        summary_parts.append("[green]0 regressions[/green]")
    if report.improvements:
        summary_parts.append(f"[green]{report.improvements} improvement(s)[/green]")
    if report.changed:
        summary_parts.append(f"[yellow]{report.changed} changed[/yellow]")
    summary_parts.append(f"[dim]{report.unchanged} unchanged[/dim]")

    con.print()
    con.print(Panel(
        " | ".join(summary_parts),
        title=f"[bold]Diff: {report.baseline_run_id} -> {report.current_run_id}[/bold]",
        expand=False,
    ))

    # --- Added / removed cases ---
    if report.added_cases:
        con.print(f"\n  [cyan]+ {len(report.added_cases)} new case(s):[/cyan] "
                   + ", ".join(report.added_cases[:5])
                   + ("..." if len(report.added_cases) > 5 else ""))
    if report.removed_cases:
        con.print(f"  [yellow]- {len(report.removed_cases)} removed case(s):[/yellow] "
                   + ", ".join(report.removed_cases[:5])
                   + ("..." if len(report.removed_cases) > 5 else ""))

    # --- Per-case diffs ---
    diffs_to_show = report.case_diffs
    if regressions_only:
        diffs_to_show = [d for d in diffs_to_show if d.status == "regression"]

    for diff in diffs_to_show:
        if diff.status == "unchanged" and not regressions_only:
            continue  # skip unchanged in normal mode too to reduce noise
        _render_case_diff(diff, con)


def _render_case_diff(diff: CaseDiff, con: Console) -> None:
    """Render a single case diff."""
    status_label = _STATUS_ICONS.get(diff.status, diff.status)
    con.print(f"\n  {status_label} [bold]{diff.query}[/bold]")

    # Triage
    if diff.triage:
        triage_label = _TRIAGE_ICONS.get(diff.triage, str(diff.triage))
        con.print(f"    Triage: {triage_label}")

    # Metrics
    _render_metrics(diff, con)

    # Judge diff
    _render_judge_diff(diff, con)

    # Context diff
    _render_context_diff(diff, con)


def _render_metrics(diff: CaseDiff, con: Console) -> None:
    """Render latency and token metrics."""
    latency_color = "red" if diff.latency_delta_pct > 50 else (
        "green" if diff.latency_delta_pct < -10 else "dim"
    )
    sign = "+" if diff.latency_delta_pct >= 0 else ""
    con.print(
        f"    Latency: {diff.latency_before:.0f}ms -> {diff.latency_after:.0f}ms "
        f"[{latency_color}]({sign}{diff.latency_delta_pct:.0f}%)[/{latency_color}]"
    )

    if diff.tokens_before is not None and diff.tokens_after is not None:
        token_delta = diff.tokens_after - diff.tokens_before
        sign = "+" if token_delta >= 0 else ""
        color = "red" if token_delta > 500 else "dim"
        con.print(
            f"    Tokens:  {diff.tokens_before} -> {diff.tokens_after} "
            f"[{color}]({sign}{token_delta})[/{color}]"
        )


def _render_judge_diff(diff: CaseDiff, con: Console) -> None:
    """Render judge verdict changes."""
    if diff.judge_before is None and diff.judge_after is None:
        return

    con.print("    [bold]Judge:[/bold]")
    dims = set()
    for v in (diff.judge_before or []):
        dims.add(v.dimension)
    for v in (diff.judge_after or []):
        dims.add(v.dimension)

    for dim in sorted(dims):
        before = next((v for v in (diff.judge_before or []) if v.dimension == dim), None)
        after = next((v for v in (diff.judge_after or []) if v.dimension == dim), None)

        before_str = _verdict_str(before) if before else "[dim]N/A[/dim]"
        after_str = _verdict_str(after) if after else "[dim]N/A[/dim]"
        con.print(f"      {dim}: {before_str} -> {after_str}")

        # Show reason if verdict changed
        if after and before and after.passed != before.passed:
            con.print(f"        Reason: {after.reason}")


def _verdict_str(v) -> str:
    """Format a single verdict as colored text."""
    if v.passed:
        return f"[green]PASS ({v.score:.1f})[/green]"
    return f"[red]FAIL ({v.score:.1f})[/red]"


def _render_context_diff(diff: CaseDiff, con: Console) -> None:
    """Render lost/new context hashes."""
    if not diff.contexts_lost and not diff.contexts_new:
        return

    con.print("    [bold]Contexts:[/bold]")
    for h in diff.contexts_lost:
        con.print(f"      [red]- LOST {h[:16]}...[/red]")
    for h in diff.contexts_new:
        con.print(f"      [green]+ NEW  {h[:16]}...[/green]")
