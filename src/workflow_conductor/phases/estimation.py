"""Execution time estimation — critical-path model for HyperFlow workflows.

Called from executor_selection phase after the advisor has chosen a model.
Estimates wall-clock duration by simulating the DAG phase by phase, bounded
by cluster parallelism.

Model:
  For each DAG phase (sequential):
    n_jobs   = ceil(task_count / agg_factor)   [1 for plain JOB]
    parallel = min(n_jobs, floor(vcpus / cpu_per_task))
    batches  = ceil(n_jobs / parallel)
    time     = batches * (JOB_STARTUP_S + agg_factor * task_duration_s)
  total = sum over phases
"""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workflow_conductor.models import PipelineState

logger = logging.getLogger(__name__)

# Estimated K8s Job startup overhead (image already cached in Kind/k3s).
_JOB_STARTUP_S = 3.0

# 1000 Genomes DAG phase order (lower = earlier).
_1000G_TASK_ORDER: dict[str, int] = {
    "individuals": 0,
    "individuals_merge": 1,
    "sifting": 2,
    "mutation_overlap": 3,
    "frequency": 4,
}

# Fallback durations when profiler has not measured a type.
_1000G_DEFAULT_DURATIONS: dict[str, float] = {
    "individuals": 30.0,
    "individuals_merge": 10.0,
    "sifting": 45.0,
    "mutation_overlap": 20.0,
    "frequency": 15.0,
}


def _parse_cpu(cpu_str: str) -> float:
    """Parse K8s CPU string to cores ('2' → 2.0, '500m' → 0.5)."""
    if not cpu_str:
        return 1.0
    if cpu_str.endswith("m"):
        try:
            return float(cpu_str[:-1]) / 1000.0
        except ValueError:
            return 1.0
    try:
        return float(cpu_str)
    except ValueError:
        return 1.0


def _count_tasks_by_type(processes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for proc in processes:
        t = proc.get("fun", "")
        if t:
            counts[t] = counts.get(t, 0) + 1
    return counts


def _phase_order(task_types: set[str]) -> list[str]:
    """Return task types sorted by 1000G dag order, unknowns last."""
    known = sorted(
        [t for t in task_types if t in _1000G_TASK_ORDER],
        key=lambda t: _1000G_TASK_ORDER[t],
    )
    unknown = sorted(t for t in task_types if t not in _1000G_TASK_ORDER)
    return known + unknown


def estimate_execution_time(state: PipelineState) -> float:
    """Return estimated wall-clock duration in seconds, or 0.0 if not enough data."""
    processes: list[dict[str, Any]] = list(
        (state.workflow_json or {}).get("processes", [])
    )
    if not processes:
        return 0.0

    available_vcpus = state.infrastructure.available_vcpus if state.infrastructure else 0
    if available_vcpus <= 0:
        available_vcpus = 4  # safe fallback for dry-run / tests

    # Build lookup tables from resource profiles
    duration_by_type: dict[str, float] = {}
    cpu_by_type: dict[str, float] = {}
    profile_by_type = {p.task_type: p for p in state.resource_profiles}
    for task_type, profile in profile_by_type.items():
        if profile.expected_duration_seconds > 0:
            duration_by_type[task_type] = profile.expected_duration_seconds
        cpu_by_type[task_type] = _parse_cpu(profile.cpu_limit)

    rec = state.execution_model_recommendation
    agg_factor = 1
    if rec and rec.agglomeration_configs:
        agg_factor = rec.agglomeration_configs[0].size

    task_counts = _count_tasks_by_type(processes)
    ordered_types = _phase_order(set(task_counts.keys()))

    total_seconds = 0.0
    for task_type in ordered_types:
        count = task_counts[task_type]
        duration = duration_by_type.get(
            task_type, _1000G_DEFAULT_DURATIONS.get(task_type, 30.0)
        )
        cpu = cpu_by_type.get(task_type, 1.0)

        max_concurrent_jobs = max(1, int(available_vcpus / cpu))

        if agg_factor > 1:
            n_jobs = math.ceil(count / agg_factor)
            parallel = min(n_jobs, max_concurrent_jobs)
            batches = math.ceil(n_jobs / parallel)
            phase_time = batches * (_JOB_STARTUP_S + agg_factor * duration)
        else:
            parallel = min(count, max_concurrent_jobs)
            batches = math.ceil(count / parallel)
            phase_time = batches * (_JOB_STARTUP_S + duration)

        logger.debug(
            "Estimation: type=%s count=%d agg=%d parallel=%d batches=%d phase_time=%.0fs",
            task_type, count, agg_factor, parallel, batches, phase_time,
        )
        total_seconds += phase_time

    return total_seconds
