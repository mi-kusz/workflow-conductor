"""Unit tests for deployment phase — simplified to kubectl cp + signal.

After the data preparation refactor, deployment only:
- Copies workflow.json to engine pod via kubectl cp
- Signals engine to start by touching .conductor-ready
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from execution_model_advisor import Advisor

from workflow_conductor.config import ConductorSettings
from workflow_conductor.models import (
    InfrastructureMeasurements,
    PipelineState,
    WorkflowPlan,
)


def _make_state(**overrides: object) -> PipelineState:
    """Create a state as if provisioning, data prep, and generation have run."""
    defaults = dict(
        workflow_plan=WorkflowPlan(description="test"),
        namespace="wf-1000g-20260216-120000",
        cluster_ready=True,
        engine_pod_name="engine-pod-123",
        infrastructure=InfrastructureMeasurements(
            namespace="wf-1000g-20260216-120000",
            node_count=2,
            available_vcpus=4,
            memory_gb=8.0,
        ),
        workflow_json={"name": "test-workflow", "processes": [], "signals": []},
    )
    defaults.update(overrides)
    return PipelineState(**defaults)  # type: ignore[arg-type]


class TestDeploymentPhase:
    @pytest.mark.asyncio
    async def test_deployment_copies_workflow_and_signals(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state()
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            result = await run_deployment_phase(state, settings)

        # workflow.json only (no columns.txt or pop files)
        kubectl.cp_to_pod.assert_called_once()
        kubectl.exec_in_pod.assert_called_once()
        # Verify signal command
        signal_call = kubectl.exec_in_pod.call_args
        assert signal_call.args[1] == ["touch", "/work_dir/.conductor-ready"]
        assert result.workflow_json_path != ""
        assert result.helm_release_name == "hf-run"

    @pytest.mark.asyncio
    async def test_deployment_requires_namespace(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(namespace="", cluster_ready=True)
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with pytest.raises(ValueError, match="namespace"):
            await run_deployment_phase(state, settings)

    @pytest.mark.asyncio
    async def test_deployment_requires_cluster_ready(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(cluster_ready=False)
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with pytest.raises(ValueError, match="cluster"):
            await run_deployment_phase(state, settings)

    @pytest.mark.asyncio
    async def test_deployment_requires_engine_pod(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(engine_pod_name="")
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with pytest.raises(ValueError, match="engine pod"):
            await run_deployment_phase(state, settings)

    @pytest.mark.asyncio
    async def test_deployment_injects_workflow_json(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(
            workflow_json={"name": "test-workflow", "processes": [], "signals": []},
        )
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            result = await run_deployment_phase(state, settings)

        kubectl.cp_to_pod.assert_called_once()
        cp_call = kubectl.cp_to_pod.call_args
        assert cp_call.args[2] == "/work_dir/workflow.json"
        assert result.workflow_json_path != ""

    @pytest.mark.asyncio
    async def test_deployment_copies_columns_txt(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(
            columns_txt="#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096",
        )
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            await run_deployment_phase(state, settings)

        # workflow.json + columns.txt = 2 cp_to_pod calls
        assert kubectl.cp_to_pod.call_count == 2
        cp_calls = kubectl.cp_to_pod.call_args_list
        destinations = [call.args[2] for call in cp_calls]
        assert "/work_dir/workflow.json" in destinations
        assert "/work_dir/columns.txt" in destinations

    @pytest.mark.asyncio
    async def test_deployment_copies_population_files(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(
            columns_txt="#CHROM\tPOS\tFORMAT\tHG00096",
            population_files={"GBR": "HG00096\nHG00097", "FIN": "HG00171"},
        )
        settings = ConductorSettings(
            kubernetes={"cluster_provider": "existing"},
        )

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            await run_deployment_phase(state, settings)

        # workflow.json + columns.txt + 2 population files = 4 cp_to_pod calls
        assert kubectl.cp_to_pod.call_count == 4
        cp_calls = kubectl.cp_to_pod.call_args_list
        destinations = [call.args[2] for call in cp_calls]
        assert "/work_dir/workflow.json" in destinations
        assert "/work_dir/columns.txt" in destinations
        assert "/work_dir/GBR" in destinations
        assert "/work_dir/FIN" in destinations


class TestDeploymentWithExecutionModel:
    def _make_large_workflow(self) -> dict:
        processes = [{"name": "individuals", "ins": [f"in{i}"], "outs": [f"out{i}"]} for i in range(2500)]
        processes += [{"name": "merge", "ins": ["x"], "outs": ["y"]} for _ in range(160)]
        return {"processes": processes}

    @pytest.mark.asyncio
    async def test_deploys_workflow_config_json_for_agglomeration(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        wf = {"processes": [{"name": "task", "ins": [f"i{i}"], "outs": [f"o{i}"]} for i in range(700)]}
        rec = Advisor.analyze(wf, available_vcpus=4)
        assert rec.model.value == "JOB_AGGLOMERATION"

        state = _make_state(
            workflow_json=wf,
            execution_model_recommendation=rec,
        )
        settings = ConductorSettings(kubernetes={"cluster_provider": "existing"})

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            await run_deployment_phase(state, settings)

        destinations = [call.args[2] for call in kubectl.cp_to_pod.call_args_list]
        assert "/work_dir/workflow.config.json" in destinations
        assert "/work_dir/workflow.json" in destinations

    @pytest.mark.asyncio
    async def test_deploys_workflow_config_json_for_worker_pool(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        wf = self._make_large_workflow()
        rec = Advisor.analyze(wf, available_vcpus=4)
        assert rec.model.value == "WORKER_POOL"

        state = _make_state(workflow_json=wf, execution_model_recommendation=rec)
        settings = ConductorSettings(kubernetes={"cluster_provider": "existing"})

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            await run_deployment_phase(state, settings)

        destinations = [call.args[2] for call in kubectl.cp_to_pod.call_args_list]
        assert "/work_dir/workflow.config.json" in destinations

    @pytest.mark.asyncio
    async def test_no_workflow_config_json_for_job_model(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        wf = {"processes": [{"name": "task", "ins": ["i"], "outs": ["o"]} for _ in range(50)]}
        rec = Advisor.analyze(wf, available_vcpus=4)
        assert rec.model.value == "JOB"

        state = _make_state(workflow_json=wf, execution_model_recommendation=rec)
        settings = ConductorSettings(kubernetes={"cluster_provider": "existing"})

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            await run_deployment_phase(state, settings)

        destinations = [call.args[2] for call in kubectl.cp_to_pod.call_args_list]
        assert "/work_dir/workflow.config.json" not in destinations

    @pytest.mark.asyncio
    async def test_no_workflow_config_json_when_no_recommendation(self) -> None:
        from workflow_conductor.phases.deployment import run_deployment_phase

        state = _make_state(execution_model_recommendation=None)
        settings = ConductorSettings(kubernetes={"cluster_provider": "existing"})

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock()
            kubectl.exec_in_pod = AsyncMock()

            await run_deployment_phase(state, settings)

        destinations = [call.args[2] for call in kubectl.cp_to_pod.call_args_list]
        assert "/work_dir/workflow.config.json" not in destinations

    @pytest.mark.asyncio
    async def test_workflow_config_json_deployed_before_signal(self) -> None:
        """workflow.config.json must be copied before .conductor-ready is touched."""
        from workflow_conductor.phases.deployment import run_deployment_phase

        wf = {"processes": [{"name": "task", "ins": [f"i{i}"], "outs": [f"o{i}"]} for i in range(700)]}
        rec = Advisor.analyze(wf, available_vcpus=4)
        call_order: list[str] = []

        state = _make_state(workflow_json=wf, execution_model_recommendation=rec)
        settings = ConductorSettings(kubernetes={"cluster_provider": "existing"})

        async def cp_side(src: str, pod: str, dest: str, **kw: object) -> None:
            call_order.append(f"cp:{dest}")

        async def exec_side(pod: str, cmd: list[str], **kw: object) -> str:
            call_order.append(f"exec:{cmd[0]}")
            return ""

        with patch("workflow_conductor.phases.deployment.Kubectl") as MockKubectl:
            kubectl = MockKubectl.return_value
            kubectl.cp_to_pod = AsyncMock(side_effect=cp_side)
            kubectl.exec_in_pod = AsyncMock(side_effect=exec_side)
            await run_deployment_phase(state, settings)

        config_idx = next(i for i, e in enumerate(call_order) if "workflow.config.json" in e)
        signal_idx = next(i for i, e in enumerate(call_order) if "touch" in e)
        assert config_idx < signal_idx
