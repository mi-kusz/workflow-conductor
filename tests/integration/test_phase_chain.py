"""Integration tests: state flows correctly between pipeline phases.

Each test runs two consecutive phase functions with mocked external
dependencies, verifying that the output of one phase is used correctly
as input to the next.  No Kind cluster required.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflow_conductor.config import ConductorSettings
from workflow_conductor.models import (
    PipelineState,
    ResourceProfile,
    WorkflowPlan,
)

# ---------------------------------------------------------------------------
# Shared mock builders
# ---------------------------------------------------------------------------

_DISPLAY_ALL = [
    "workflow_conductor.phases.provisioning.display_phase_header",
    "workflow_conductor.phases.data_preparation.display_phase_header",
    "workflow_conductor.phases.monitoring.display_phase_header",
    "workflow_conductor.phases.monitoring.display_sentinel_banner",
    "workflow_conductor.phases.monitoring.display_report",
    "workflow_conductor.phases.monitoring.display_completion_summary",
    "workflow_conductor.phases.completion.display_phase_header",
    "workflow_conductor.phases.completion.display_completion_summary",
]


def _mock_kubectl(**overrides: Any) -> MagicMock:
    m = MagicMock()
    m.create_namespace = AsyncMock(return_value="")
    m.create_resource_quota = AsyncMock()
    m.cleanup_previous_runs = AsyncMock()
    m.wait_for_pod = AsyncMock(return_value="engine-pod-abc")
    m.wait_for_job = AsyncMock()
    m.get_nodes = AsyncMock(
        return_value={
            "node_count": 1,
            "total_cpu": 4,
            "total_memory_gb": 8.0,
            "k8s_version": "v1.31.0",
        }
    )
    m.exec_in_pod = AsyncMock(return_value="")
    m.apply_json = AsyncMock()
    m.delete_namespace = AsyncMock()
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


def _mock_helm() -> MagicMock:
    m = MagicMock()
    m.upgrade_install = AsyncMock()
    m.release_exists = AsyncMock(return_value=True)
    m.uninstall = AsyncMock()
    return m


def _mock_kind(exists: bool = True) -> MagicMock:
    m = MagicMock()
    m.exists = AsyncMock(return_value=exists)
    m.create = AsyncMock()
    m.load_image = AsyncMock()
    m.export_kubeconfig = AsyncMock(return_value="/tmp/kube.yaml")
    m.name = "test-cluster"
    return m


def _mock_sentinel(summary: Any) -> MagicMock:
    import asyncio

    m = MagicMock()
    m.watch = AsyncMock(return_value=summary)
    m.reports = asyncio.Queue()
    m.kubectl = MagicMock()
    m.kubectl.logs = AsyncMock(return_value="")
    return m


def _make_watch_summary(**kwargs: Any) -> Any:
    from execution_sentinel.models import WatchSummary

    return WatchSummary(**kwargs)


def _patch_all(*names: str) -> list[Any]:
    """Convenience: return a list of patch context managers."""
    return [patch(n) for n in names]


# ---------------------------------------------------------------------------
# Provisioning → Data Preparation
# ---------------------------------------------------------------------------


class TestProvisioningToDataPreparation:
    async def test_data_prep_uses_namespace_from_provisioning(self) -> None:
        """Namespace from provisioning forwards to data_prep."""
        from workflow_conductor.phases.data_preparation import (
            run_data_preparation_phase,
        )
        from workflow_conductor.phases.provisioning import run_provisioning_phase

        settings = ConductorSettings()
        state = PipelineState(
            user_prompt="Analyze EUR chr1",
            workflow_plan=WorkflowPlan(
                chromosomes=["1"],
                populations=["EUR"],
                parallelism=10,
                estimated_data_size_gb=0.1,
                download_commands=[
                    "tabix http://ftp.example.com/ALL.chr1.phase3.vcf.gz 1:1-100000 > /work_dir/ALL.chr1.250000.vcf"
                ],
            ),
        )

        # ── Phase 4: Provisioning ───────────────────────────────────────────
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.provisioning.Kubectl",
                    return_value=_mock_kubectl(),
                )
            )
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.provisioning.Helm",
                    return_value=_mock_helm(),
                )
            )
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.provisioning.KindCluster",
                    return_value=_mock_kind(),
                )
            )
            stack.enter_context(
                patch("workflow_conductor.phases.provisioning.display_phase_header")
            )
            state = await run_provisioning_phase(state, settings)

        provisioned_ns = state.namespace
        provisioned_pod = state.engine_pod_name
        assert provisioned_ns.startswith("wf-1000g-")
        assert provisioned_pod == "engine-pod-abc"

        # ── Phase 5: Data Preparation ───────────────────────────────────────
        # exec_in_pod call 1: discovery → chr data, call 2: header extraction → ""
        discovery_output = "1:1234:ALL.chr1.250000.vcf:none"
        exec_calls: list[dict[str, Any]] = []

        async def exec_side(
            pod: str, _cmd: list[str], *, namespace: str, **kw: Any
        ) -> str:
            exec_calls.append({"pod": pod, "namespace": namespace})
            return discovery_output if len(exec_calls) == 1 else ""

        kubectl2 = _mock_kubectl(exec_in_pod=AsyncMock(side_effect=exec_side))
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.data_preparation.Kubectl",
                    return_value=kubectl2,
                )
            )
            stack.enter_context(
                patch("workflow_conductor.phases.data_preparation.display_phase_header")
            )
            state = await run_data_preparation_phase(state, settings)

        # Verify all exec calls used the namespace/pod from provisioning
        for c in exec_calls:
            assert c["namespace"] == provisioned_ns
            assert c["pod"] == provisioned_pod

        assert len(state.chromosome_data) == 1
        assert state.chromosome_data[0].chromosome == "1"
        assert state.chromosome_data[0].row_count == 1234

    async def test_data_prep_raises_without_provisioned_namespace(self) -> None:
        """run_data_preparation_phase raises ValueError when namespace is not set."""
        from workflow_conductor.phases.data_preparation import (
            run_data_preparation_phase,
        )

        state = PipelineState(
            engine_pod_name="engine-0",
            namespace="",  # provisioning has not run
            workflow_plan=WorkflowPlan(chromosomes=["1"], populations=["EUR"]),
        )
        with pytest.raises(ValueError, match="namespace not set"):
            await run_data_preparation_phase(state, ConductorSettings())

    async def test_data_prep_raises_without_engine_pod(self) -> None:
        """data_preparation raises when engine_pod_name unset."""
        from workflow_conductor.phases.data_preparation import (
            run_data_preparation_phase,
        )

        state = PipelineState(
            namespace="wf-1000g-20260101",
            engine_pod_name="",  # provisioning has not run
            workflow_plan=WorkflowPlan(chromosomes=["1"], populations=["EUR"]),
        )
        with pytest.raises(ValueError, match="engine pod not set"):
            await run_data_preparation_phase(state, ConductorSettings())

    async def test_data_prep_parses_multiple_chromosomes(self) -> None:
        """row counts parsed for each chromosome individually."""
        from workflow_conductor.phases.data_preparation import (
            run_data_preparation_phase,
        )

        state = PipelineState(
            engine_pod_name="engine-0",
            namespace="wf-ns",
            workflow_plan=WorkflowPlan(
                chromosomes=["1", "2", "3"],
                populations=["EUR"],
                download_commands=[
                    "tabix http://ftp.example.com/ALL.chr1.phase3.vcf.gz > /work_dir/ALL.chr1.250000.vcf",
                    "tabix http://ftp.example.com/ALL.chr2.phase3.vcf.gz > /work_dir/ALL.chr2.250000.vcf",
                    "tabix http://ftp.example.com/ALL.chr3.phase3.vcf.gz > /work_dir/ALL.chr3.250000.vcf",
                ],
            ),
        )
        discovery = (
            "1:1000:ALL.chr1.250000.vcf:ALL.chr1.annotation.vcf\n"
            "2:2000:ALL.chr2.250000.vcf:none\n"
            "3:3000:ALL.chr3.250000.vcf:ALL.chr3.annotation.vcf"
        )
        call_n = 0

        async def exec_side(_pod: str, _cmd: list[str], **kw: Any) -> str:
            nonlocal call_n
            call_n += 1
            return discovery if call_n == 1 else ""

        kubectl = _mock_kubectl(exec_in_pod=AsyncMock(side_effect=exec_side))
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.data_preparation.Kubectl",
                    return_value=kubectl,
                )
            )
            stack.enter_context(
                patch("workflow_conductor.phases.data_preparation.display_phase_header")
            )
            state = await run_data_preparation_phase(state, ConductorSettings())

        assert len(state.chromosome_data) == 3
        by_chrom = {c.chromosome: c for c in state.chromosome_data}
        assert by_chrom["1"].row_count == 1000
        assert by_chrom["2"].row_count == 2000
        assert by_chrom["3"].row_count == 3000
        assert by_chrom["1"].annotation_file == "ALL.chr1.annotation.vcf"
        assert by_chrom["2"].annotation_file == ""  # "none" → empty string

    async def test_data_prep_skips_malformed_discovery_lines(self) -> None:
        """Lines with fewer than 3 colon-separated parts are silently skipped."""
        from workflow_conductor.phases.data_preparation import (
            run_data_preparation_phase,
        )

        state = PipelineState(
            engine_pod_name="engine-0",
            namespace="wf-ns",
            workflow_plan=WorkflowPlan(
                chromosomes=["1", "2"],
                populations=["EUR"],
                download_commands=[
                    "tabix http://ftp.example.com/ALL.chr1.phase3.vcf.gz > /work_dir/ALL.chr1.vcf",
                    "tabix http://ftp.example.com/ALL.chr2.phase3.vcf.gz > /work_dir/ALL.chr2.vcf",
                ],
            ),
        )
        discovery = (
            "malformed-line\n"
            "1:1000:ALL.chr1.vcf:none\n"
            "also:bad\n"
            "2:999:ALL.chr2.vcf:none"
        )
        call_n = 0

        async def exec_side(_pod: str, _cmd: list[str], **kw: Any) -> str:
            nonlocal call_n
            call_n += 1
            return discovery if call_n == 1 else ""

        kubectl = _mock_kubectl(exec_in_pod=AsyncMock(side_effect=exec_side))
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.data_preparation.Kubectl",
                    return_value=kubectl,
                )
            )
            stack.enter_context(
                patch("workflow_conductor.phases.data_preparation.display_phase_header")
            )
            state = await run_data_preparation_phase(state, ConductorSettings())

        # Only valid lines parsed; malformed ones skipped
        assert len(state.chromosome_data) == 2


# ---------------------------------------------------------------------------
# Monitoring → Completion state handoff
# ---------------------------------------------------------------------------


class TestMonitoringToCompletion:
    async def test_completed_workflow_produces_completed_status(self) -> None:
        """exit_code=0 → status COMPLETED."""
        from workflow_conductor.models import PipelineStatus
        from workflow_conductor.phases.completion import run_completion_phase
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        settings = ConductorSettings(auto_teardown=False)
        state = PipelineState(
            namespace="wf-1000g-test",
            engine_pod_name="engine-0",
            total_task_count=10,
        )

        # ── Phase 9: Monitoring ─────────────────────────────────────────────
        summary = _make_watch_summary(completed_tasks=10, total_tasks=10, exit_code=0)
        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _mock_sentinel(summary)
            state = await run_monitoring_phase(state, settings)

        assert state.workflow_status == "completed"
        assert state.task_completion_count == 10

        # ── Phase 10: Completion ────────────────────────────────────────────
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Kubectl",
                    return_value=_mock_kubectl(),
                )
            )
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Helm",
                    return_value=_mock_helm(),
                )
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            state = await run_completion_phase(state, settings)

        assert state.status == PipelineStatus.COMPLETED
        assert state.execution_summary is not None
        assert state.execution_summary.completed_tasks == 10
        assert state.execution_summary.total_tasks == 10

    async def test_failed_workflow_produces_failed_status(self) -> None:
        """exit_code=1 → status FAILED."""
        from workflow_conductor.models import PipelineStatus
        from workflow_conductor.phases.completion import run_completion_phase
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        settings = ConductorSettings(auto_teardown=False)
        state = PipelineState(
            namespace="wf-1000g-test",
            engine_pod_name="engine-0",
        )

        summary = _make_watch_summary(
            completed_tasks=2, failed_tasks=3, total_tasks=5, exit_code=1
        )
        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _mock_sentinel(summary)
            state = await run_monitoring_phase(state, settings)

        assert state.workflow_status == "failed"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Kubectl",
                    return_value=_mock_kubectl(),
                )
            )
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Helm",
                    return_value=_mock_helm(),
                )
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            state = await run_completion_phase(state, settings)

        assert state.status == PipelineStatus.FAILED

    async def test_timeout_produces_failed_status(self) -> None:
        """Sentinel timeout → workflow_status='timeout' → PipelineStatus.FAILED."""
        from workflow_conductor.models import PipelineStatus
        from workflow_conductor.phases.completion import run_completion_phase
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        settings = ConductorSettings(auto_teardown=False)
        state = PipelineState(
            namespace="wf-1000g-test",
            engine_pod_name="engine-0",
        )

        summary = _make_watch_summary(timed_out=True)
        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _mock_sentinel(summary)
            state = await run_monitoring_phase(state, settings)

        assert state.workflow_status == "timeout"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Kubectl",
                    return_value=_mock_kubectl(),
                )
            )
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Helm",
                    return_value=_mock_helm(),
                )
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            state = await run_completion_phase(state, settings)

        assert state.status == PipelineStatus.FAILED

    async def test_task_counts_flow_into_execution_summary(self) -> None:
        """Task counts from monitoring phase appear in ExecutionSummary."""
        from workflow_conductor.phases.completion import run_completion_phase
        from workflow_conductor.phases.monitoring import run_monitoring_phase

        settings = ConductorSettings(auto_teardown=False)
        state = PipelineState(
            namespace="wf-1000g-test",
            engine_pod_name="engine-0",
        )

        summary = _make_watch_summary(completed_tasks=7, total_tasks=10, exit_code=0)
        with ExitStack() as stack:
            MockSentinel = stack.enter_context(
                patch("workflow_conductor.phases.monitoring.Sentinel")
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            MockSentinel.return_value = _mock_sentinel(summary)
            state = await run_monitoring_phase(state, settings)

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Kubectl",
                    return_value=_mock_kubectl(),
                )
            )
            stack.enter_context(
                patch(
                    "workflow_conductor.phases.completion.Helm",
                    return_value=_mock_helm(),
                )
            )
            for p in _DISPLAY_ALL:
                stack.enter_context(patch(p))
            state = await run_completion_phase(state, settings)

        esummary = state.execution_summary
        assert esummary is not None
        assert esummary.completed_tasks == 7
        assert esummary.total_tasks == 10
        assert esummary.failed_tasks == 3


# ---------------------------------------------------------------------------
# Monitoring context construction (_build_monitoring_context)
# ---------------------------------------------------------------------------


class TestBuildMonitoringContext:
    def test_task_inventory_built_from_workflow_processes(self) -> None:
        """Task inventory reflects the processes list in workflow_json."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="wf-ns",
            engine_pod_name="engine-0",
            workflow_json={
                "processes": [
                    {"name": "individuals-0", "fun": "selectVariants"},
                    {"name": "individuals-1", "fun": "selectVariants"},
                    {"name": "merge-0", "fun": "mergeVariants"},
                ]
            },
        )

        ctx = _build_monitoring_context(state)

        assert ctx.namespace == "wf-ns"
        assert ctx.engine_pod_name == "engine-0"
        assert len(ctx.task_inventory) == 3
        assert ctx.task_inventory[0].name == "individuals-0"
        assert ctx.task_inventory[0].task_type == "selectVariants"
        assert ctx.task_inventory[2].task_type == "mergeVariants"

    def test_resource_profiles_applied_to_matching_task_types(self) -> None:
        """Resource profiles are matched by task_type and applied to TaskSpec."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="wf-ns",
            engine_pod_name="engine-0",
            workflow_json={
                "processes": [
                    {"name": "task-0", "fun": "selectVariants"},
                    {"name": "task-1", "fun": "mergeVariants"},
                ]
            },
            resource_profiles=[
                ResourceProfile(
                    task_type="selectVariants",
                    memory_limit="512Mi",
                    cpu_limit="500m",
                ),
            ],
        )

        ctx = _build_monitoring_context(state)

        sv = next(t for t in ctx.task_inventory if t.task_type == "selectVariants")
        mv = next(t for t in ctx.task_inventory if t.task_type == "mergeVariants")

        assert sv.memory_limit == "512Mi"
        assert sv.cpu_limit == "500m"
        assert mv.memory_limit == ""  # no matching profile
        assert mv.cpu_limit == ""

    def test_no_workflow_json_gives_empty_inventory(self) -> None:
        """workflow_json=None → empty task_inventory."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="wf-ns",
            engine_pod_name="engine-0",
            workflow_json=None,
        )

        ctx = _build_monitoring_context(state)
        assert ctx.task_inventory == []

    def test_workflow_json_without_processes_key(self) -> None:
        """workflow_json without 'processes' key → empty task_inventory."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="wf-ns",
            engine_pod_name="engine-0",
            workflow_json={"name": "1000genome", "signals": []},
        )

        ctx = _build_monitoring_context(state)
        assert ctx.task_inventory == []

    def test_engine_container_is_hyperflow(self) -> None:
        """engine_container is always 'hyperflow'."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        state = PipelineState(
            namespace="ns",
            engine_pod_name="pod-0",
            workflow_json={"processes": []},
        )

        ctx = _build_monitoring_context(state)
        assert ctx.engine_container == "hyperflow"

    def test_large_process_list_creates_matching_inventory(self) -> None:
        """All N processes become N TaskSpec entries."""
        from workflow_conductor.phases.monitoring import _build_monitoring_context

        n = 50
        state = PipelineState(
            namespace="ns",
            engine_pod_name="pod-0",
            workflow_json={
                "processes": [
                    {"name": f"task-{i}", "fun": "selectVariants"} for i in range(n)
                ]
            },
        )

        ctx = _build_monitoring_context(state)
        assert len(ctx.task_inventory) == n
        assert ctx.task_inventory[0].name == "task-0"
        assert ctx.task_inventory[n - 1].name == f"task-{n - 1}"
