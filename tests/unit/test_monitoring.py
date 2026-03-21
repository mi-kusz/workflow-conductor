"""Unit tests for monitoring phase — mocked Sentinel."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflow_conductor.config import ConductorSettings
from workflow_conductor.models import PipelineState, ResourceProfile

if TYPE_CHECKING:
    from execution_sentinel.models import WatchSummary


def _make_summary(
    completed: int = 0,
    failed: int = 0,
    total: int = 0,
    exit_code: int | None = None,
    timed_out: bool = False,
    mass_failure: bool = False,
) -> WatchSummary:
    from execution_sentinel.models import WatchSummary

    return WatchSummary(
        completed_tasks=completed,
        failed_tasks=failed,
        total_tasks=total,
        exit_code=exit_code,
        timed_out=timed_out,
        mass_failure=mass_failure,
    )


def _make_sentinel_mock(summary: WatchSummary) -> MagicMock:
    """Build a mock Sentinel instance that returns the given summary."""
    instance = MagicMock()
    instance.watch = AsyncMock(return_value=summary)
    instance.reports = asyncio.Queue()
    instance.kubectl = MagicMock()
    instance.kubectl.logs = AsyncMock(return_value="")
    return instance


_DISPLAY_PATCHES = [
    "workflow_conductor.phases.monitoring.display_phase_header",
    "workflow_conductor.phases.monitoring.display_sentinel_banner",
    "workflow_conductor.phases.monitoring.display_report",
    "workflow_conductor.phases.monitoring.display_completion_summary",
]


class TestMonitoringPhase:
    @pytest.mark.asyncio
    async def test_detects_workflow_completion(self) -> None:
        """Sentinel returns exit_code=0 → workflow 'completed'."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(completed=5, total=5, exit_code=0)

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _make_sentinel_mock(summary)
            result = await run_monitoring_phase(state, settings)

        assert result.workflow_status == "completed"
        assert result.task_completion_count == 5
        assert result.total_task_count == 5

    @pytest.mark.asyncio
    async def test_detects_workflow_failure(self) -> None:
        """Sentinel returns exit_code=1 → workflow 'failed'."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(completed=2, failed=1, total=5, exit_code=1)

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _make_sentinel_mock(summary)
            result = await run_monitoring_phase(state, settings)

        assert result.workflow_status == "failed"

    @pytest.mark.asyncio
    async def test_detects_mass_failure(self) -> None:
        """Sentinel sets mass_failure=True → workflow 'failed'."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(completed=1, failed=4, total=5, mass_failure=True)

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _make_sentinel_mock(summary)
            result = await run_monitoring_phase(state, settings)

        assert result.workflow_status == "failed"

    @pytest.mark.asyncio
    async def test_detects_timeout(self) -> None:
        """Sentinel sets timed_out=True → workflow 'timeout'."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(timed_out=True)

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _make_sentinel_mock(summary)
            result = await run_monitoring_phase(state, settings)

        assert result.workflow_status == "timeout"

    @pytest.mark.asyncio
    async def test_requires_engine_pod(self) -> None:
        """Missing engine_pod_name → ValueError before Sentinel is created."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="")
        settings = ConductorSettings()

        with pytest.raises(ValueError, match="engine pod not set"):
            await run_monitoring_phase(state, settings)

    @pytest.mark.asyncio
    async def test_task_counts_from_summary(self) -> None:
        """completed_tasks and total_tasks are mapped from WatchSummary."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(completed=3, total=7, exit_code=0)

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _make_sentinel_mock(summary)
            result = await run_monitoring_phase(state, settings)

        assert result.task_completion_count == 3
        assert result.total_task_count == 7

    @pytest.mark.asyncio
    async def test_sentinel_constructed_with_correct_namespace(self) -> None:
        """Sentinel receives the namespace from PipelineState."""
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="my-namespace", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(exit_code=0)

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            mock_instance = _make_sentinel_mock(summary)
            MockSentinel.return_value = mock_instance
            await run_monitoring_phase(state, settings)

        context_arg = MockSentinel.call_args[0][0]
        assert context_arg.namespace == "my-namespace"

    @pytest.mark.asyncio
    async def test_accumulates_llm_usage(self) -> None:
        """Monitoring phase accumulates LLM token usage from _translate_to_nl."""
        from execution_sentinel.models import ReportKind, SentinelReport

        from workflow_conductor.phases.monitoring import run_monitoring_phase

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-pod-1")
        settings = ConductorSettings()
        summary = _make_summary(completed=5, total=5, exit_code=0)

        mock_sentinel = _make_sentinel_mock(summary)

        # Pre-fill the queue with two reports
        r1 = SentinelReport(
            kind=ReportKind.PROGRESS, message="2/5", completed_tasks=2, total_tasks=5
        )
        r2 = SentinelReport(
            kind=ReportKind.COMPLETION, message="5/5", completed_tasks=5, total_tasks=5
        )
        mock_sentinel.reports.put_nowait(r1)
        mock_sentinel.reports.put_nowait(r2)

        # Make watch() yield control so the drain task can process queued reports
        async def _slow_watch() -> _make_summary.__class__:  # type: ignore[name-defined]
            while not mock_sentinel.reports.empty():
                await asyncio.sleep(0.01)
            return summary

        mock_sentinel.watch = _slow_watch

        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_PATCHES:
                stack.enter_context(patch(p))
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.monitoring._translate_to_nl",
                    AsyncMock(
                        return_value=(
                            "Translated.",
                            {
                                "input_tokens": 30,
                                "output_tokens": 10,
                                "model": "claude-test",
                            },
                        )
                    ),
                )
            )
            MockSentinel.return_value = mock_sentinel
            result = await run_monitoring_phase(state, settings)

        assert "monitoring" in result.llm_usage
        assert result.llm_usage["monitoring"]["input_tokens"] == 60
        assert result.llm_usage["monitoring"]["output_tokens"] == 20
        assert result.llm_usage["monitoring"]["api_calls"] == 2
        assert result.llm_usage["monitoring"]["model"] == "claude-test"


# ---------------------------------------------------------------------------
# Test: _build_monitoring_context
# ---------------------------------------------------------------------------


class TestBuildMonitoringContext:
    def test_dag_order_for_known_task_types(self) -> None:
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json={
                "processes": [
                    {"name": "sifting-0", "fun": "sifting"},
                    {"name": "frequency-0", "fun": "frequency"},
                    {"name": "individuals-0", "fun": "individuals"},
                ]
            },
        )
        ctx = _build_monitoring_context(state)
        by_type = {t.task_type: t.dag_order for t in ctx.task_inventory}
        assert by_type["individuals"] == 0
        assert by_type["sifting"] == 2
        assert by_type["frequency"] == 4

    def test_unknown_task_type_gets_dynamic_order(self) -> None:
        """Unknown task types fall back to dag_order computed from signal topology.
        A single disconnected process has depth 0."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json={"processes": [{"name": "custom-0", "fun": "custom_task"}]},
        )
        ctx = _build_monitoring_context(state)
        # Not in _1000G_TASK_ORDER → dynamic order (0 for a standalone process)
        assert ctx.task_inventory[0].dag_order == 0

    def test_unknown_task_type_with_no_fun_gets_order_99(self) -> None:
        """Process with missing/empty 'fun' field cannot be resolved → dag_order 99."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json={"processes": [{"name": "custom-0", "fun": ""}]},
        )
        ctx = _build_monitoring_context(state)
        assert ctx.task_inventory[0].dag_order == 99

    def test_sets_dag_structure(self) -> None:
        from workflow_conductor.phases.monitoring import (
            _1000G_TASK_DEPS,
            _build_monitoring_context,
        )

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-0")
        ctx = _build_monitoring_context(state)
        assert ctx.dag_structure == _1000G_TASK_DEPS

    def test_uses_profiler_duration_when_set(self) -> None:
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json={"processes": [{"name": "sifting-0", "fun": "sifting"}]},
            resource_profiles=[
                ResourceProfile(task_type="sifting", expected_duration_seconds=120.0)
            ],
        )
        ctx = _build_monitoring_context(state)
        assert ctx.task_inventory[0].expected_duration_seconds == 120.0

    def test_falls_back_to_default_duration(self) -> None:
        from workflow_conductor.phases.monitoring import (
            _1000G_DEFAULT_DURATIONS,
            _build_monitoring_context,
        )

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json={"processes": [{"name": "sifting-0", "fun": "sifting"}]},
        )
        ctx = _build_monitoring_context(state)
        assert (
            ctx.task_inventory[0].expected_duration_seconds
            == _1000G_DEFAULT_DURATIONS["sifting"]
        )

    def test_profiler_zero_duration_uses_default(self) -> None:
        """Profiler entry with duration=0 falls back to default."""
        from workflow_conductor.phases.monitoring import (
            _1000G_DEFAULT_DURATIONS,
            _build_monitoring_context,
        )

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json={"processes": [{"name": "sifting-0", "fun": "sifting"}]},
            resource_profiles=[
                ResourceProfile(task_type="sifting", expected_duration_seconds=0.0)
            ],
        )
        ctx = _build_monitoring_context(state)
        assert (
            ctx.task_inventory[0].expected_duration_seconds
            == _1000G_DEFAULT_DURATIONS["sifting"]
        )

    def test_empty_workflow_json_empty_inventory(self) -> None:
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-0")
        ctx = _build_monitoring_context(state)
        assert ctx.task_inventory == []

    def test_agglomeration_factor_from_recommendation(self) -> None:
        """agglomeration_factor is taken from the first AgglomerationConfig.size."""
        from execution_model_advisor import Advisor

        from workflow_conductor.phases.monitoring import _build_monitoring_context

        # Build a workflow large enough to trigger JOB_AGGLOMERATION.
        # Advisor groups by "name" field, so all 1000 processes must share the same name.
        wf = {
            "processes": [
                {"name": "sifting", "fun": "sifting", "ins": [], "outs": []}
                for _ in range(1000)
            ]
        }
        rec = Advisor.analyze(wf, available_vcpus=4)
        assert rec.agglomeration_configs, "expected agglomeration recommendation"

        state = PipelineState(
            namespace="test-ns",
            engine_pod_name="engine-0",
            workflow_json=wf,
            execution_model_recommendation=rec,
        )
        ctx = _build_monitoring_context(state)
        assert ctx.agglomeration_factor == rec.agglomeration_configs[0].size
        assert ctx.execution_model == rec.model.value

    def test_agglomeration_factor_default_when_no_recommendation(self) -> None:
        """Without an advisor recommendation agglomeration_factor defaults to 1."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(namespace="test-ns", engine_pod_name="engine-0")
        ctx = _build_monitoring_context(state)
        assert ctx.agglomeration_factor == 1
        assert ctx.execution_model == "JOB"


# ---------------------------------------------------------------------------
# Test: _translate_to_nl
# ---------------------------------------------------------------------------


class TestTranslateToNl:
    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self) -> None:
        """On API error, _translate_to_nl returns message unchanged."""
        from execution_sentinel.models import ReportKind, SentinelReport

        from workflow_conductor.phases.monitoring import _translate_to_nl

        report = SentinelReport(
            kind=ReportKind.PROGRESS,
            message="3/10 tasks completed.",
            completed_tasks=3,
            total_tasks=10,
        )
        settings = ConductorSettings()

        with patch(
            "anthropic.AsyncAnthropic", side_effect=Exception("API unavailable")
        ):
            text, usage = await _translate_to_nl(report, settings)

        assert text == "3/10 tasks completed."
        assert usage == {}

    @pytest.mark.asyncio
    async def test_returns_stripped_translated_text(self) -> None:
        """Successful API call returns stripped translated text."""
        from execution_sentinel.models import ReportKind, SentinelReport

        from workflow_conductor.phases.monitoring import _translate_to_nl

        report = SentinelReport(
            kind=ReportKind.PROGRESS,
            message="3/10 tasks completed.",
            completed_tasks=3,
            total_tasks=10,
        )
        settings = ConductorSettings()

        mock_usage = MagicMock()
        mock_usage.input_tokens = 50
        mock_usage.output_tokens = 20
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  Analysis is 30% complete.  ")]
        mock_response.usage = mock_usage
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            text, usage = await _translate_to_nl(report, settings)

        assert text == "Analysis is 30% complete."
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 20
        assert usage["model"] == settings.llm.anthropic_model

    @pytest.mark.asyncio
    async def test_uses_configured_model(self) -> None:
        """_translate_to_nl passes settings.llm.anthropic_model to the API."""
        from execution_sentinel.models import ReportKind, SentinelReport

        from workflow_conductor.phases.monitoring import _translate_to_nl

        report = SentinelReport(kind=ReportKind.COMPLETION, message="Done.")
        settings = ConductorSettings()

        mock_usage = MagicMock()
        mock_usage.input_tokens = 40
        mock_usage.output_tokens = 15
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Workflow finished.")]
        mock_response.usage = mock_usage
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            await _translate_to_nl(report, settings)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == settings.llm.anthropic_model


# ---------------------------------------------------------------------------
# Test: _capture_cluster_snapshot
# ---------------------------------------------------------------------------


class TestClusterSnapshots:
    @pytest.mark.asyncio
    async def test_capture_cluster_snapshot(self) -> None:
        from workflow_conductor.phases.monitoring import _capture_cluster_snapshot

        mock_kubectl = AsyncMock()
        mock_kubectl._run = AsyncMock(
            side_effect=[
                "node-4  500m  12%  2Gi  45%",
                "job-1  100m  256Mi",
            ]
        )

        snapshot = await _capture_cluster_snapshot(mock_kubectl, "test-ns")
        assert "timestamp" in snapshot
        assert "nodes" in snapshot
        assert "pods" in snapshot

    @pytest.mark.asyncio
    async def test_capture_snapshot_handles_errors(self) -> None:
        from workflow_conductor.phases.monitoring import _capture_cluster_snapshot

        mock_kubectl = AsyncMock()
        mock_kubectl._run = AsyncMock(side_effect=Exception("metrics not available"))

        snapshot = await _capture_cluster_snapshot(mock_kubectl, "test-ns")
        assert snapshot["nodes"] == ""
        assert snapshot["pods"] == ""
