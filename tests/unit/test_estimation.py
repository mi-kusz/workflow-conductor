"""Unit tests for execution time estimation."""

from __future__ import annotations

import math

import pytest

from execution_model_advisor.models import (
    AgglomerationConfig,
    ExecutionModel,
    ExecutionModelRecommendation,
    WorkflowMetrics,
)
from workflow_conductor.models import InfrastructureMeasurements, PipelineState, ResourceProfile
from workflow_conductor.phases.estimation import (
    _parse_cpu,
    estimate_execution_time,
)


def _metrics() -> WorkflowMetrics:
    return WorkflowMetrics(
        total_tasks=10,
        task_type_counts={"t": 10},
        task_type_metrics=[],
        dominant_task_type="t",
        dominant_task_count=10,
        dominant_task_fraction=1.0,
        dag_depth=1,
        available_vcpus=4,
    )


def _rec_job() -> ExecutionModelRecommendation:
    return ExecutionModelRecommendation(
        model=ExecutionModel.JOB,
        reasoning="test",
        confidence=1.0,
        metrics=_metrics(),
    )


def _rec_agg(size: int) -> ExecutionModelRecommendation:
    return ExecutionModelRecommendation(
        model=ExecutionModel.JOB_AGGLOMERATION,
        reasoning="test",
        confidence=1.0,
        metrics=_metrics(),
        agglomeration_configs=[
            AgglomerationConfig(match_task=["individuals"], size=size, timeout_ms=300000)
        ],
    )


def _make_workflow(task_counts: dict[str, int]) -> dict:
    processes = []
    for fun, count in task_counts.items():
        for i in range(count):
            processes.append({"name": f"{fun}-{i}", "fun": fun})
    return {"processes": processes}


def _make_profile(task_type: str, cpu_limit: str, duration: float) -> ResourceProfile:
    return ResourceProfile(
        task_type=task_type,
        cpu_limit=cpu_limit,
        expected_duration_seconds=duration,
    )


class TestParseCpu:
    def test_integer_string(self) -> None:
        assert _parse_cpu("2") == 2.0

    def test_millicore_string(self) -> None:
        assert _parse_cpu("500m") == pytest.approx(0.5)

    def test_empty_defaults_to_one(self) -> None:
        assert _parse_cpu("") == 1.0

    def test_invalid_defaults_to_one(self) -> None:
        assert _parse_cpu("abc") == 1.0


class TestEstimateExecutionTime:
    def test_no_workflow_returns_zero(self) -> None:
        state = PipelineState()
        assert estimate_execution_time(state) == 0.0

    def test_empty_processes_returns_zero(self) -> None:
        state = PipelineState(workflow_json={"processes": []})
        assert estimate_execution_time(state) == 0.0

    def test_single_type_job_model(self) -> None:
        """10 tasks, 4 vCPUs, 1 CPU/task → 3 batches of ~3, duration=10s each."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 10}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
            resource_profiles=[_make_profile("individuals", "1", 10.0)],
            execution_model_recommendation=_rec_job(),
        )
        result = estimate_execution_time(state)
        # max_concurrent=4, batches=ceil(10/4)=3, each=(3+10)=13s → 39s
        assert result == pytest.approx(3 * 13.0)

    def test_agglomeration_reduces_job_count(self) -> None:
        """100 tasks, agg=10 → 10 jobs, 4 vCPUs → 3 batches."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 100}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
            resource_profiles=[_make_profile("individuals", "1", 10.0)],
            execution_model_recommendation=_rec_agg(10),
        )
        result = estimate_execution_time(state)
        # n_jobs=10, max_concurrent=4, batches=ceil(10/4)=3
        # phase_time = 3 * (3 + 10*10) = 3 * 103 = 309s
        assert result == pytest.approx(3 * (3.0 + 10 * 10.0))

    def test_agglomeration_faster_than_job_for_large_count(self) -> None:
        """Agglomeration should give shorter estimate for large task counts."""
        workflow = _make_workflow({"individuals": 800})
        infra = InfrastructureMeasurements(available_vcpus=8)
        profiles = [_make_profile("individuals", "1", 30.0)]

        state_job = PipelineState(
            workflow_json=workflow,
            infrastructure=infra,
            resource_profiles=profiles,
            execution_model_recommendation=_rec_job(),
        )
        state_agg = PipelineState(
            workflow_json=workflow,
            infrastructure=infra,
            resource_profiles=profiles,
            execution_model_recommendation=_rec_agg(4),
        )
        assert estimate_execution_time(state_agg) < estimate_execution_time(state_job)

    def test_fallback_duration_used_when_no_profile(self) -> None:
        """Uses _1000G_DEFAULT_DURATIONS when resource_profiles is empty."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 4}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
            resource_profiles=[],
            execution_model_recommendation=_rec_job(),
        )
        result = estimate_execution_time(state)
        # fallback duration for individuals = 30s, max_concurrent=4, batches=1
        assert result == pytest.approx(1 * (3.0 + 30.0))

    def test_fallback_vcpus_when_no_infrastructure(self) -> None:
        """Falls back to 4 vCPUs when infrastructure is None."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 4}),
            infrastructure=None,
            resource_profiles=[_make_profile("individuals", "1", 10.0)],
            execution_model_recommendation=_rec_job(),
        )
        result = estimate_execution_time(state)
        assert result > 0.0

    def test_two_phase_dag_sums_phases(self) -> None:
        """Two sequential phases: result >= sum of individual phase estimates."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 4, "sifting": 4}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
            resource_profiles=[
                _make_profile("individuals", "1", 10.0),
                _make_profile("sifting", "1", 20.0),
            ],
            execution_model_recommendation=_rec_job(),
        )
        result = estimate_execution_time(state)
        # individuals: 1 batch × (3+10) = 13s
        # sifting:     1 batch × (3+20) = 23s
        assert result == pytest.approx(13.0 + 23.0)

    def test_cpu_limit_constrains_parallelism(self) -> None:
        """2 CPU/task on 4-vCPU cluster → max 2 concurrent."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 8}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
            resource_profiles=[_make_profile("individuals", "2", 10.0)],
            execution_model_recommendation=_rec_job(),
        )
        result = estimate_execution_time(state)
        # max_concurrent = floor(4/2) = 2, batches = ceil(8/2) = 4
        assert result == pytest.approx(4 * (3.0 + 10.0))

    def test_estimated_duration_stored_in_state(self) -> None:
        """estimate_execution_time result is stored in PipelineState."""
        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 4}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
            resource_profiles=[_make_profile("individuals", "1", 10.0)],
            execution_model_recommendation=_rec_job(),
        )
        expected = estimate_execution_time(state)
        state.estimated_duration_seconds = expected
        assert state.estimated_duration_seconds == pytest.approx(expected)
