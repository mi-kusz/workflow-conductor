"""Unit tests for models.py."""

from __future__ import annotations

import json

from workflow_conductor.models import (
    ChromosomeData,
    ExecutionSummary,
    InfrastructureMeasurements,
    IntentClassification,
    PhaseResult,
    PipelineError,
    PipelinePhase,
    PipelineState,
    PipelineStatus,
    ResourceProfile,
    RetryPolicy,
    UserResponse,
    WorkflowPlan,
)


class TestEnums:
    def test_pipeline_phases(self) -> None:
        assert PipelinePhase.ROUTING.value == "routing"
        assert PipelinePhase.PROVISIONING.value == "provisioning"
        assert PipelinePhase.DATA_PREPARATION.value == "data_preparation"
        assert PipelinePhase.GENERATION.value == "generation"
        assert PipelinePhase.APPROVAL.value == "approval"
        assert PipelinePhase.COMPLETION.value == "completion"
        assert len(PipelinePhase) == 11

    def test_pipeline_status(self) -> None:
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.ABORTED.value == "aborted"
        assert len(PipelineStatus) == 6


class TestPhaseResult:
    def test_defaults(self) -> None:
        result = PhaseResult(
            phase=PipelinePhase.ROUTING,
            status=PipelineStatus.COMPLETED,
        )
        assert result.data == {}
        assert result.error is None
        assert result.duration_seconds == 0.0

    def test_with_error(self) -> None:
        result = PhaseResult(
            phase=PipelinePhase.DEPLOYMENT,
            status=PipelineStatus.FAILED,
            error="Helm timeout",
        )
        assert result.error == "Helm timeout"


class TestWorkflowPlan:
    def test_defaults(self) -> None:
        plan = WorkflowPlan()
        assert plan.chromosomes == []
        assert plan.populations == []
        assert plan.parallelism is None
        assert plan.estimated_data_size_gb == 0.0

    def test_no_shared_mutable_defaults(self) -> None:
        """Two instances must not share the same list objects."""
        plan_a = WorkflowPlan()
        plan_b = WorkflowPlan()
        plan_a.chromosomes.append("22")
        assert plan_b.chromosomes == []

    def test_json_roundtrip(self) -> None:
        plan = WorkflowPlan(
            chromosomes=["1", "22"],
            populations=["EUR"],
            parallelism=5,
            description="Test plan",
        )
        data = json.loads(plan.model_dump_json())
        restored = WorkflowPlan.model_validate(data)
        assert restored.chromosomes == ["1", "22"]
        assert restored.parallelism == 5


class TestSupportingModels:
    def test_execution_summary_defaults(self) -> None:
        summary = ExecutionSummary()
        assert summary.total_tasks == 0
        assert summary.output_path == ""

    def test_pipeline_error(self) -> None:
        err = PipelineError(
            phase="deployment",
            error_type="HelmTimeout",
            message="Timed out after 15m",
        )
        assert err.recoverable is True
        assert err.suggested_action == ""

    def test_intent_classification(self) -> None:
        intent = IntentClassification(
            analysis_type="variant_frequency",
            populations=["EUR", "EAS"],
            chromosomes=[1, 22],
        )
        assert intent.focus == ""

    def test_user_response(self) -> None:
        resp = UserResponse(action="refine", feedback="Use less parallelism")
        assert resp.action == "refine"

    def test_resource_profile(self) -> None:
        profile = ResourceProfile(
            task_type="individuals",
            cpu_request="500m",
            memory_request="512Mi",
            source="profiler",
        )
        assert profile.confidence == 0.0

    def test_resource_profile_expected_duration_default(self) -> None:
        profile = ResourceProfile(task_type="individuals")
        assert profile.expected_duration_seconds == 0.0

    def test_resource_profile_with_expected_duration(self) -> None:
        profile = ResourceProfile(task_type="sifting", expected_duration_seconds=45.0)
        assert profile.expected_duration_seconds == 45.0

    def test_infrastructure_measurements(self) -> None:
        infra = InfrastructureMeasurements(
            namespace="wf-test",
            node_count=3,
            available_vcpus=8,
        )
        assert infra.actual_data_sizes == {}


class TestPipelineState:
    def test_defaults(self) -> None:
        state = PipelineState()
        assert state.current_phase == PipelinePhase.ROUTING
        assert state.status == PipelineStatus.PENDING
        assert state.phase_results == []
        assert state.conversation_history == []

    def test_no_shared_mutable_defaults(self) -> None:
        """Two instances must not share list/dict objects."""
        state_a = PipelineState()
        state_b = PipelineState()
        state_a.phase_results.append(
            PhaseResult(
                phase=PipelinePhase.ROUTING,
                status=PipelineStatus.COMPLETED,
            )
        )
        assert state_b.phase_results == []

    def test_record_phase(self) -> None:
        state = PipelineState()
        result = PhaseResult(
            phase=PipelinePhase.ROUTING,
            status=PipelineStatus.COMPLETED,
            duration_seconds=1.5,
        )
        state.record_phase(result)
        assert len(state.phase_results) == 1
        assert state.phase_timings["routing"] == 1.5

    def test_record_phase_zero_duration_not_stored(self) -> None:
        state = PipelineState()
        result = PhaseResult(
            phase=PipelinePhase.ROUTING,
            status=PipelineStatus.COMPLETED,
            duration_seconds=0.0,
        )
        state.record_phase(result)
        assert len(state.phase_results) == 1
        assert "routing" not in state.phase_timings

    def test_add_conversation(self) -> None:
        state = PipelineState()
        state.add_conversation("user", "Analyze EUR chr22")
        assert len(state.conversation_history) == 1
        assert state.conversation_history[0]["role"] == "user"
        assert state.conversation_history[0]["content"] == "Analyze EUR chr22"
        assert "timestamp" in state.conversation_history[0]

    def test_synthesize_context_basic(self) -> None:
        state = PipelineState(user_prompt="Analyze EUR chr22")
        ctx = state.synthesize_context_for_composer()
        assert "Analyze EUR chr22" in ctx

    def test_synthesize_context_with_plan_and_feedback(self) -> None:
        state = PipelineState(
            user_prompt="Analyze EUR chr22",
            workflow_plan=WorkflowPlan(
                chromosomes=["22"],
                populations=["EUR"],
                parallelism=5,
                description="EUR chr22 analysis",
            ),
            user_modifications="Use higher parallelism",
        )
        ctx = state.synthesize_context_for_composer()
        assert "Analyze EUR chr22" in ctx
        assert "EUR chr22 analysis" in ctx
        assert "Use higher parallelism" in ctx

    def test_synthesize_context_with_infrastructure(self) -> None:
        state = PipelineState(
            user_prompt="Test",
            infrastructure=InfrastructureMeasurements(
                node_count=3,
                available_vcpus=8,
                memory_gb=16.0,
            ),
        )
        ctx = state.synthesize_context_for_composer()
        assert "3 nodes" in ctx
        assert "8 vCPUs" in ctx

    def test_json_roundtrip(self) -> None:
        state = PipelineState(
            execution_id="wf-1000g-20260213-abc123",
            user_prompt="Analyze EUR chr22",
            current_phase=PipelinePhase.PLANNING,
            status=PipelineStatus.IN_PROGRESS,
            workflow_plan=WorkflowPlan(
                chromosomes=["22"],
                populations=["EUR"],
            ),
        )
        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)
        assert restored.execution_id == "wf-1000g-20260213-abc123"
        assert restored.current_phase == PipelinePhase.PLANNING
        assert restored.workflow_plan is not None
        assert restored.workflow_plan.chromosomes == ["22"]

    def test_model_dump_produces_dict(self) -> None:
        state = PipelineState(execution_id="test-123")
        dumped = state.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["execution_id"] == "test-123"

    def test_chromosome_data_default_empty(self) -> None:
        state = PipelineState()
        assert state.chromosome_data == []

    def test_chromosome_data_no_shared_mutable(self) -> None:
        state_a = PipelineState()
        state_b = PipelineState()
        state_a.chromosome_data.append(
            ChromosomeData(
                vcf_file="ALL.chr22.vcf",
                row_count=1000,
                annotation_file="ALL.chr22.annotation.vcf",
                chromosome="22",
            )
        )
        assert state_b.chromosome_data == []


class TestChromosomeData:
    def test_required_fields(self) -> None:
        cd = ChromosomeData(
            vcf_file="ALL.chr22.vcf",
            row_count=494328,
            annotation_file="ALL.chr22.annotation.vcf",
            chromosome="22",
        )
        assert cd.vcf_file == "ALL.chr22.vcf"
        assert cd.row_count == 494328
        assert cd.chromosome == "22"

    def test_json_roundtrip(self) -> None:
        cd = ChromosomeData(
            vcf_file="ALL.chr6.hla.vcf",
            row_count=12345,
            annotation_file="ALL.chr6.hla.annotation.vcf",
            chromosome="6",
        )
        data = json.loads(cd.model_dump_json())
        restored = ChromosomeData.model_validate(data)
        assert restored.vcf_file == "ALL.chr6.hla.vcf"
        assert restored.row_count == 12345


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_base == 2.0
        assert policy.retryable_errors == []

    def test_custom_values(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            backoff_base=1.5,
            retryable_errors=["KubectlError", "HelmError"],
        )
        assert policy.max_retries == 5
        assert len(policy.retryable_errors) == 2


class TestExperimentDataFields:
    def test_llm_usage_default_empty(self) -> None:
        state = PipelineState()
        assert state.llm_usage == {}

    def test_planning_estimates_default_empty(self) -> None:
        state = PipelineState()
        assert state.planning_estimates == {}

    def test_cluster_snapshots_default_empty(self) -> None:
        state = PipelineState()
        assert state.cluster_snapshots == []

    def test_experiment_fields_serialize(self) -> None:
        state = PipelineState()
        state.llm_usage = {"planning": {"input_tokens": 100, "output_tokens": 50}}
        state.planning_estimates = {"estimated_tasks": 340}
        state.cluster_snapshots = [{"timestamp": "2026-03-12T14:00:00", "nodes": "..."}]
        data = state.model_dump()
        assert data["llm_usage"]["planning"]["input_tokens"] == 100
        assert data["planning_estimates"]["estimated_tasks"] == 340
        assert len(data["cluster_snapshots"]) == 1
        restored = PipelineState.from_json(state.to_json())
        assert restored.llm_usage == state.llm_usage
