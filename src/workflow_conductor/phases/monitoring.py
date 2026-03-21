"""Phase 9: Execution monitoring — watch for workflow completion signal.

Delegates all monitoring logic to the `execution-sentinel` package via
``Sentinel.watch()``.  The Sentinel detects OOMKills, stragglers,
mass-failure escalation, and connectivity loss, streaming ``SentinelReport``
objects into a queue that is drained here and rendered via Rich.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from execution_sentinel.config import SentinelSettings  # type: ignore[import-untyped]
from execution_sentinel.models import (  # type: ignore[import-untyped]
    MonitoringContext,
    SentinelReport,
    TaskSpec,
)
from execution_sentinel.sentinel import Sentinel  # type: ignore[import-untyped]
from execution_sentinel.ui.display import (  # type: ignore[import-untyped]
    display_completion_summary,
    display_report,
    display_sentinel_banner,
)

from workflow_conductor.k8s import Kubectl
from workflow_conductor.models import PipelinePhase, PipelineState
from workflow_conductor.ui.display import display_phase_header

if TYPE_CHECKING:
    from workflow_conductor.config import ConductorSettings

logger = logging.getLogger(__name__)

# Known 1000 Genomes task ordering and dependency structure.
# Used as primary source of dag_order; dynamic computation is the fallback.
_1000G_TASK_ORDER: dict[str, int] = {
    "individuals": 0,
    "individuals_merge": 1,
    "sifting": 2,
    "mutation_overlap": 3,
    "frequency": 4,
}
_1000G_TASK_DEPS: dict[str, list[str]] = {
    "individuals": [],
    "individuals_merge": ["individuals"],
    "sifting": ["individuals_merge"],
    "mutation_overlap": ["sifting"],
    "frequency": ["sifting"],
}

# Fallback durations for 1000 Genomes task types (seconds).
# Used only when the profiler has not supplied measured values.
_1000G_DEFAULT_DURATIONS: dict[str, float] = {
    "individuals": 30.0,
    "individuals_merge": 10.0,
    "sifting": 45.0,
    "mutation_overlap": 20.0,
    "frequency": 15.0,
}


def _dag_order_from_workflow(processes: list[dict]) -> dict[str, int]:
    """Compute a topological dag_order (depth) for each task type from workflow.json.

    Signals in ``outs`` of process A matched against ``ins`` of process B give
    edge A → B.  The returned dict maps task type (``fun`` field) → depth
    (0-based), so earlier phases get lower numbers.
    """
    n = len(processes)
    if n == 0:
        return {}

    signal_to_producer: dict[str, int] = {}
    for i, p in enumerate(processes):
        for sig in p.get("outs", []):
            if isinstance(sig, str):
                signal_to_producer[sig] = i

    from collections import deque
    children: list[list[int]] = [[] for _ in range(n)]
    in_degree: list[int] = [0] * n
    for i, p in enumerate(processes):
        for sig in p.get("ins", []):
            if isinstance(sig, str) and sig in signal_to_producer:
                parent = signal_to_producer[sig]
                if parent != i:
                    children[parent].append(i)
                    in_degree[i] += 1

    depth = [0] * n
    queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
    while queue:
        node = queue.popleft()
        for child in children[node]:
            depth[child] = max(depth[child], depth[node] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Map task type → minimum depth seen for that type
    type_depth: dict[str, int] = {}
    for i, p in enumerate(processes):
        t = p.get("fun", "")
        if t and (t not in type_depth or depth[i] < type_depth[t]):
            type_depth[t] = depth[i]
    return type_depth


def _build_monitoring_context(state: PipelineState) -> MonitoringContext:
    processes = list((state.workflow_json or {}).get("processes", []))
    profile_by_type = {p.task_type: p for p in state.resource_profiles}

    # Compute dag_order dynamically; fall back to flat ordering if signals are
    # not connected (all processes at depth 0 → use insertion order by type).
    dynamic_order = _dag_order_from_workflow(processes)

    # If the dynamic computation yielded no useful spread (all zeros), derive
    # ordering from the first occurrence of each task type.
    if len(set(dynamic_order.values())) <= 1 and len(dynamic_order) > 1:
        seen: dict[str, int] = {}
        counter = 0
        for p in processes:
            t = p.get("fun", "")
            if t and t not in seen:
                seen[t] = counter
                counter += 1
        dynamic_order = seen

    task_inventory = []
    for proc in processes:
        task_type = proc.get("fun", "")
        profile = profile_by_type.get(task_type)
        dag_order = _1000G_TASK_ORDER.get(task_type, dynamic_order.get(task_type, 99))
        expected_duration = (
            profile.expected_duration_seconds
            if (profile and profile.expected_duration_seconds > 0)
            else _1000G_DEFAULT_DURATIONS.get(task_type, 0.0)
        )
        task_inventory.append(
            TaskSpec(
                name=proc.get("name", ""),
                task_type=task_type,
                memory_limit=profile.memory_limit if profile else "",
                cpu_limit=profile.cpu_limit if profile else "",
                dag_order=dag_order,
                expected_duration_seconds=expected_duration,
            )
        )

    # Extract agglomeration_factor and execution_model from advisor recommendation
    rec = state.execution_model_recommendation
    agglomeration_factor = 1
    execution_model = "JOB"
    if rec is not None:
        execution_model = rec.model.value
        if rec.agglomeration_configs:
            agglomeration_factor = rec.agglomeration_configs[0].size

    return MonitoringContext(
        namespace=state.namespace,
        engine_pod_name=state.engine_pod_name,
        engine_container="hyperflow",
        task_inventory=task_inventory,
        dag_structure=_1000G_TASK_DEPS,
        agglomeration_factor=agglomeration_factor,
        execution_model=execution_model,
    )


async def _translate_to_nl(
    report: SentinelReport, settings: ConductorSettings
) -> tuple[str, dict[str, Any]]:
    """Translate a Sentinel report into a science-friendly sentence.

    Returns (translated_text, usage_dict).
    """
    try:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.AsyncAnthropic()
        prompt = (
            f"You are an AI assistant helping a genomics researcher. "
            f"Translate this Kubernetes workflow monitoring event into one clear, "
            f"science-friendly sentence (no Kubernetes jargon).\n"
            f"Event: kind={report.kind}, message={report.message!r}, "
            f"progress={report.completed_tasks}/{report.total_tasks}."
        )
        response = await client.messages.create(
            model=settings.llm.anthropic_model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        result: str = response.content[0].text.strip()  # type: ignore[union-attr]
        usage: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": settings.llm.anthropic_model,
        }
        return result, usage
    except Exception:  # noqa: BLE001
        return str(report.message), {}


async def _capture_cluster_snapshot(kubectl: Kubectl, namespace: str) -> dict[str, str]:
    """Capture a point-in-time cluster utilization snapshot via kubectl top."""
    snapshot: dict[str, str] = {
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        nodes_output = await kubectl._run(["top", "nodes", "--no-headers"])
        snapshot["nodes"] = nodes_output.strip()
    except Exception:
        snapshot["nodes"] = ""
    try:
        pods_output = await kubectl._run(
            ["top", "pods", "-n", namespace, "--no-headers"]
        )
        snapshot["pods"] = pods_output.strip()
    except Exception:
        snapshot["pods"] = ""
    return snapshot


async def run_monitoring_phase(
    state: PipelineState,
    settings: ConductorSettings,
) -> PipelineState:
    """Monitor workflow execution via Execution Sentinel.

    Delegates to ``Sentinel.watch()`` which handles OOMKill detection,
    straggler detection, mass-failure escalation, and connectivity loss.
    """
    display_phase_header(PipelinePhase.MONITORING)

    if not state.engine_pod_name:
        raise ValueError("Cannot monitor: engine pod not set")

    context = _build_monitoring_context(state)
    sentinel_settings = SentinelSettings(
        kubeconfig=settings.kubernetes.kubeconfig,
        poll_interval=settings.monitor_poll_interval,
        timeout=settings.monitor_timeout,
    )
    sentinel = Sentinel(context, sentinel_settings)

    logger.info(
        "Starting Execution Sentinel: namespace=%s, engine_pod=%s, "
        "poll=%ds, timeout=%ds, expected_tasks=%d",
        context.namespace,
        context.engine_pod_name,
        sentinel_settings.poll_interval,
        sentinel_settings.timeout,
        context.total_expected_tasks,
    )

    display_sentinel_banner(context.namespace, context.total_expected_tasks)

    monitoring_llm_usage: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "api_calls": 0,
        "model": "",
    }

    kubectl = Kubectl(kubeconfig=settings.kubernetes.kubeconfig)

    async def _drain() -> None:
        while True:
            report = await sentinel.reports.get()
            text, usage = await _translate_to_nl(report, settings)
            report.nl_message = text
            if usage:
                monitoring_llm_usage["input_tokens"] += usage.get("input_tokens", 0)
                monitoring_llm_usage["output_tokens"] += usage.get("output_tokens", 0)
                monitoring_llm_usage["api_calls"] += 1
                monitoring_llm_usage["model"] = usage.get("model", "")
            display_report(report)

    drain_task = asyncio.create_task(_drain())

    async def _snapshot_loop() -> None:
        """Capture cluster utilization every 30 seconds."""
        while True:
            await asyncio.sleep(30)
            snapshot = await _capture_cluster_snapshot(kubectl, state.namespace)
            state.cluster_snapshots.append(snapshot)

    snapshot_task = asyncio.create_task(_snapshot_loop())
    summary = await sentinel.watch()
    snapshot_task.cancel()
    drain_task.cancel()

    # Flush any reports produced in the final iteration
    while not sentinel.reports.empty():
        display_report(sentinel.reports.get_nowait())

    display_completion_summary(summary)

    # Map WatchSummary → PipelineState
    state.task_completion_count = summary.completed_tasks
    state.total_task_count = summary.total_tasks or len(context.task_inventory)
    if summary.timed_out:
        state.workflow_status = "timeout"
    elif summary.mass_failure or (
        summary.exit_code is not None and summary.exit_code != 0
    ):
        state.workflow_status = "failed"
    else:
        state.workflow_status = "completed"

    if monitoring_llm_usage["api_calls"] > 0:
        state.llm_usage["monitoring"] = dict(monitoring_llm_usage)

    # Estimation accuracy: compare predicted vs actual duration
    if state.estimated_duration_seconds > 0 and summary.duration_seconds > 0:
        error_pct = (
            (summary.duration_seconds - state.estimated_duration_seconds)
            / state.estimated_duration_seconds
            * 100
        )
        state.estimation_accuracy = {
            "estimated_s": state.estimated_duration_seconds,
            "actual_s": summary.duration_seconds,
            "error_pct": error_pct,
        }
        logger.info(
            "Estimation accuracy: predicted=%.0fs actual=%.0fs error=%+.1f%%",
            state.estimated_duration_seconds,
            summary.duration_seconds,
            error_pct,
        )

    # Capture engine logs via sentinel's kubectl client
    try:
        logs = await sentinel.kubectl.logs(
            state.engine_pod_name,
            namespace=state.namespace,
            container="hyperflow",
            tail=100,
        )
        logger.debug("Engine logs (last 100 lines):\n%s", logs)
    except Exception:
        logger.warning("Failed to capture engine logs")

    return state
