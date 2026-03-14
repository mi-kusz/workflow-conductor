"""Rich display components for pipeline phases."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from workflow_conductor.models import (
    ExecutionSummary,
    PipelinePhase,
    PipelineState,
    WorkflowPlan,
)

console = Console()


def display_pipeline_banner(
    prompt: str, *, dry_run: bool = False, demo: bool = False
) -> None:
    """Show the initial pipeline banner with the user prompt."""
    mode = ""
    if dry_run:
        mode = "[bold yellow]DRY RUN[/bold yellow] "
    elif demo:
        mode = "[bold magenta]DEMO MODE[/bold magenta] "
    console.print(
        Panel(
            f"[bold]{prompt}[/bold]",
            title=f"{mode}[bold cyan]Workflow Conductor[/bold cyan]",
            subtitle="AI-powered genomics workflow orchestrator",
            border_style="cyan",
        )
    )


def display_phase_header(phase: PipelinePhase) -> None:
    """Show a phase transition header."""
    labels = {
        PipelinePhase.ROUTING: "Phase 1: Routing",
        PipelinePhase.PLANNING: "Phase 2: Planning",
        PipelinePhase.VALIDATION: "Phase 3: Validation (Gate 1)",
        PipelinePhase.PROVISIONING: "Phase 4: Provisioning",
        PipelinePhase.DATA_PREPARATION: "Phase 5: Data Preparation",
        PipelinePhase.GENERATION: "Phase 6: Generation",
        PipelinePhase.EXECUTOR_SELECTION: "Phase 6b: Executor Selection",
        PipelinePhase.APPROVAL: "Phase 7: Approval (Gate 2)",
        PipelinePhase.DEPLOYMENT: "Phase 8: Deployment",
        PipelinePhase.MONITORING: "Phase 9: Monitoring",
        PipelinePhase.COMPLETION: "Phase 10: Completion",
    }
    label = labels.get(phase, phase.value)
    console.rule(f"[bold blue]{label}[/bold blue]")


def display_workflow_plan(plan: WorkflowPlan) -> None:
    """Display a workflow plan in a Rich panel with a summary table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    if plan.description:
        table.add_row("Description", plan.description)
    if plan.populations:
        table.add_row("Populations", ", ".join(plan.populations))
    if plan.chromosomes:
        table.add_row("Chromosomes", ", ".join(plan.chromosomes))
    if plan.parallelism is not None:
        table.add_row("Parallelism", str(plan.parallelism))
    if plan.estimated_data_size_gb > 0:
        table.add_row("Est. Data Size", f"{plan.estimated_data_size_gb:.1f} GB")

    console.print(
        Panel(
            table, title="[bold green]Workflow Plan[/bold green]", border_style="green"
        )
    )


def display_completion_summary(state: PipelineState) -> None:
    """Display the completion summary after pipeline finishes."""
    summary: ExecutionSummary | None = state.execution_summary

    table = Table(title="Execution Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Execution ID", state.execution_id or "N/A")
    table.add_row("Final Status", state.status.value)
    table.add_row("Namespace", state.namespace or "N/A")

    if summary:
        table.add_row("Total Tasks", str(summary.total_tasks))
        table.add_row("Completed", str(summary.completed_tasks))
        table.add_row("Failed", str(summary.failed_tasks))
        if summary.total_runtime_seconds > 0:
            mins = summary.total_runtime_seconds / 60
            table.add_row("Runtime", f"{mins:.1f} min")

    if state.phase_timings:
        for phase_name, duration in state.phase_timings.items():
            table.add_row(f"  {phase_name}", f"{duration:.1f}s")

    console.print(Panel(table, border_style="blue"))


def display_execution_preview(state: PipelineState) -> None:
    """Display execution preview with real task counts and resource info."""
    import json
    import re

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    if state.workflow_json:
        # workflow.json size
        json_bytes = len(json.dumps(state.workflow_json).encode())
        if json_bytes >= 1024 * 1024:
            size_str = f"{json_bytes / (1024 * 1024):.1f} MB"
        elif json_bytes >= 1024:
            size_str = f"{json_bytes / 1024:.0f} KB"
        else:
            size_str = f"{json_bytes} B"
        table.add_row("Workflow JSON Size", size_str)

        processes = state.workflow_json.get("processes", [])
        table.add_row("Total Processes", str(len(processes)))

        # Count by task type prefix and collect chromosomes
        type_counts: dict[str, int] = {}
        chromosomes: set[str] = set()
        for proc in processes:
            name = proc.get("name", "")
            # Extract type from name like "individuals_chr22_1" → "individuals"
            task_type = name.split("_")[0] if "_" in name else name
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
            # Extract chromosome from name like "individuals_chr22_1"
            chr_match = re.search(r"chr(\d+|[XY])", name)
            if chr_match:
                chromosomes.add(chr_match.group(1))
        for task_type, count in sorted(type_counts.items()):
            table.add_row(f"  {task_type}", str(count))

        signals = state.workflow_json.get("signals", [])
        table.add_row("Signals", str(len(signals)))

        if chromosomes:
            sorted_chrs = sorted(
                chromosomes,
                key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c),
            )
            table.add_row("Chromosomes", ", ".join(sorted_chrs))

    if state.infrastructure:
        table.add_row("Nodes", str(state.infrastructure.node_count))
        table.add_row("vCPUs", str(state.infrastructure.available_vcpus))
        table.add_row("Memory", f"{state.infrastructure.memory_gb:.1f} GB")

    if state.namespace:
        table.add_row("Namespace", state.namespace)

    console.print(
        Panel(
            table,
            title="[bold yellow]Execution Preview[/bold yellow]",
            border_style="yellow",
        )
    )


def display_error(message: str, *, phase: str = "") -> None:
    """Display an error message."""
    prefix = f"\\[{phase}] " if phase else ""
    console.print(f"[bold red]{prefix}{message}[/bold red]")


# --- Demo mode display functions ---

_PHASE_EXPLANATIONS: dict[PipelinePhase, str] = {
    PipelinePhase.ROUTING: (
        "Classifies the natural-language prompt to select the right "
        "workflow type. Currently routes all genomics prompts to the "
        "1000 Genomes pipeline."
    ),
    PipelinePhase.PLANNING: (
        "The LLM calls the Workflow Composer MCP server to analyze the "
        "research question and create a workflow plan.\n"
        "Watch for: plan_workflow and estimate_variants tool calls."
    ),
    PipelinePhase.VALIDATION: (
        "Gate 1: presents the workflow plan for review. In auto-approve "
        "mode the plan is accepted automatically. Otherwise the user can "
        "approve, refine, or abort."
    ),
    PipelinePhase.PROVISIONING: (
        "Sets up Kubernetes infrastructure: creates namespace, deploys "
        "hf-ops (NFS server, Redis), installs hf-run (engine + workers), "
        "and waits for data container staging to complete."
    ),
    PipelinePhase.DATA_PREPARATION: (
        "Decompresses and stages data files on the NFS volume. "
        "Scans VCF files for exact variant row counts to feed "
        "into the workflow generator for precise parallelism."
    ),
    PipelinePhase.GENERATION: (
        "The LLM calls the Workflow Composer to generate the full "
        "workflow.json — the HyperFlow DAG of processes and signals.\n"
        "Watch for: generate_workflow tool call."
    ),
    PipelinePhase.EXECUTOR_SELECTION: (
        "Analyzes workflow.json metrics (task counts, homogeneity) and "
        "selects the optimal HyperFlow execution model: JOB, "
        "JOB_AGGLOMERATION, or WORKER_POOL. Generates workflow.config.json "
        "if a non-default model is chosen."
    ),
    PipelinePhase.APPROVAL: (
        "Gate 2: shows an execution preview with real task counts and "
        "resource info. In auto-approve mode the execution is approved "
        "automatically."
    ),
    PipelinePhase.DEPLOYMENT: (
        "Delivers the generated workflow.json to the engine pod via "
        "kubectl cp and signals the engine to start by touching "
        ".conductor-ready."
    ),
    PipelinePhase.MONITORING: (
        "Polls Kubernetes for job completion status. Tracks completed "
        "and failed jobs until all tasks finish or a timeout is reached."
    ),
    PipelinePhase.COMPLETION: (
        "Builds the execution summary and tears down the namespace. "
        "In demo mode, the previous pause is your window to inspect "
        "the cluster before cleanup."
    ),
}


def display_phase_explanation(phase: PipelinePhase) -> None:
    """Show a demo-mode explanation panel for a pipeline phase."""
    text = _PHASE_EXPLANATIONS.get(phase, phase.value)
    console.print(
        Panel(
            text,
            title=f"[bold magenta]{phase.value.title()}[/bold magenta]",
            border_style="magenta",
        )
    )


def display_workflow_json_summary(workflow_json: dict[str, Any]) -> None:
    """Show a summary of the generated workflow.json."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    processes: list[Any] = workflow_json.get("processes", [])
    table.add_row("Total Processes", str(len(processes)))

    # Count by task type prefix
    type_counts: dict[str, int] = {}
    for proc in processes:
        name = proc.get("name", "")
        task_type = name.split("_")[0] if "_" in name else name
        type_counts[task_type] = type_counts.get(task_type, 0) + 1
    for task_type, count in sorted(type_counts.items()):
        table.add_row(f"  {task_type}", str(count))

    signals: list[Any] = workflow_json.get("signals", [])
    table.add_row("Signals", str(len(signals)))

    console.print(
        Panel(
            table,
            title="[bold green]Workflow JSON Summary[/bold green]",
            border_style="green",
        )
    )


def demo_pause(message: str = "Press Enter to continue...") -> None:
    """Block until the presenter presses Enter."""
    console.input(f"\n[dim]{message}[/dim]")
