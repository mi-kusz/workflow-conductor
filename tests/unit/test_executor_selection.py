"""Unit tests for executor_selection phase."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from workflow_conductor.config import ConductorSettings
from workflow_conductor.models import InfrastructureMeasurements, PipelineState


def _make_workflow(task_counts: dict[str, int]) -> dict:
    processes = []
    for name, count in task_counts.items():
        for i in range(count):
            processes.append({"name": name, "ins": [f"in{i}"], "outs": [f"out{i}"]})
    return {"processes": processes}


def _settings() -> ConductorSettings:
    return ConductorSettings(kubernetes={"cluster_provider": "existing"})


class TestExecutorSelectionPhase:
    @pytest.mark.asyncio
    async def test_small_workflow_selects_job(self) -> None:
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(
            workflow_json=_make_workflow({"taskA": 50, "taskB": 20}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
        )
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        assert result.execution_model_recommendation is not None
        assert result.execution_model_recommendation.model.value == "JOB"
        assert result.execution_model_recommendation.hyperflow_config == {}

    @pytest.mark.asyncio
    async def test_large_homogeneous_selects_agglomeration(self) -> None:
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 2500, "split": 80, "merge": 80}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
        )
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        rec = result.execution_model_recommendation
        assert rec is not None
        assert rec.model.value == "JOB_AGGLOMERATION"
        assert rec.hyperflow_config["jobAgglomerations"][0]["matchTask"] == ["individuals"]

    @pytest.mark.asyncio
    async def test_medium_large_selects_agglomeration(self) -> None:
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(
            workflow_json=_make_workflow({"compute": 700, "pre": 50}),
            infrastructure=InfrastructureMeasurements(available_vcpus=4),
        )
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        rec = result.execution_model_recommendation
        assert rec is not None
        assert rec.model.value == "JOB_AGGLOMERATION"
        assert "jobAgglomerations" in rec.hyperflow_config

    @pytest.mark.asyncio
    async def test_no_workflow_json_skips_gracefully(self) -> None:
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(workflow_json=None)
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        assert result.execution_model_recommendation is None

    @pytest.mark.asyncio
    async def test_vcpus_taken_from_infrastructure(self) -> None:
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(
            workflow_json=_make_workflow({"individuals": 2500, "other": 160}),
            infrastructure=InfrastructureMeasurements(available_vcpus=16),
        )
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        assert result.execution_model_recommendation is not None
        assert result.execution_model_recommendation.metrics.available_vcpus == 16

    @pytest.mark.asyncio
    async def test_no_infrastructure_defaults_vcpus_to_zero(self) -> None:
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(
            workflow_json=_make_workflow({"task": 50}),
            infrastructure=None,
        )
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        assert result.execution_model_recommendation is not None
        assert result.execution_model_recommendation.metrics.available_vcpus == 0

    @pytest.mark.asyncio
    async def test_recommendation_stored_in_state(self) -> None:
        from execution_model_advisor.models import ExecutionModelRecommendation
        from workflow_conductor.phases.executor_selection import run_executor_selection_phase

        state = PipelineState(workflow_json=_make_workflow({"task": 50}))
        with patch("workflow_conductor.phases.executor_selection.display_phase_header"):
            result = await run_executor_selection_phase(state, _settings())

        assert isinstance(result.execution_model_recommendation, ExecutionModelRecommendation)
