"""All data models for the Workflow Conductor pipeline.

All models use Pydantic BaseModel. JSON serialization via .model_dump()
and .model_dump_json(). All timestamps use datetime.now(UTC).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from execution_model_advisor.models import ExecutionModelRecommendation


class PipelinePhase(StrEnum):
    """Pipeline phases — 10-phase architecture.

    Phases PROVISIONING, DATA_PREPARATION, GENERATION, and APPROVAL
    between VALIDATION and DEPLOYMENT.
    """

    ROUTING = "routing"
    PLANNING = "planning"
    VALIDATION = "validation"
    PROVISIONING = "provisioning"
    DATA_PREPARATION = "data_preparation"
    GENERATION = "generation"
    EXECUTOR_SELECTION = "executor_selection"
    APPROVAL = "approval"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETION = "completion"


class PipelineStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# --- Phase & Plan Models ---


class PhaseResult(BaseModel):
    """Outcome of a single phase execution."""

    phase: PipelinePhase
    status: PipelineStatus
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0


class WorkflowPlan(BaseModel):
    """Plan from Workflow Composer (Phase 2 output)."""

    chromosomes: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    parallelism: int | None = None
    estimated_data_size_gb: float = 0.0
    description: str = ""
    download_commands: list[str] = Field(default_factory=list)
    data_preparation: dict[str, Any] = Field(default_factory=dict)
    raw_plan: dict[str, Any] = Field(default_factory=dict)


class InfrastructureMeasurements(BaseModel):
    """Measurements from infrastructure provisioning (Stage 2+)."""

    namespace: str = ""
    actual_data_sizes: dict[str, int] = Field(default_factory=dict)
    total_data_size_mb: float = 0.0
    available_vcpus: int = 0
    node_count: int = 0
    memory_gb: float = 0.0
    k8s_version: str = ""
    storage_class: str = ""


class ChromosomeData(BaseModel):
    """Per-chromosome data for workflow generation.

    Feeds into the Composer's generate_workflow tool. Row counts are
    exact — scanned from VCF files during data preparation.
    """

    vcf_file: str
    row_count: int
    annotation_file: str
    chromosome: str
    file_size_bytes: int = 0


class ResourceProfile(BaseModel):
    """Per-task-type resource recommendations."""

    task_type: str = ""
    cpu_request: str = ""
    cpu_limit: str = ""
    memory_request: str = ""
    memory_limit: str = ""
    confidence: float = 0.0
    source: str = ""  # "profiler" or "composer"
    expected_duration_seconds: float = 0.0  # filled by profiler or fallback


# --- Supporting Models ---


class ExecutionSummary(BaseModel):
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_runtime_seconds: float = 0.0
    output_path: str = ""


class PipelineError(BaseModel):
    phase: str
    error_type: str
    message: str
    timestamp: str = ""
    recoverable: bool = True
    suggested_action: str = ""


class RetryPolicy(BaseModel):
    """Configuration for phase retry behavior."""

    max_retries: int = 3
    backoff_base: float = 2.0
    retryable_errors: list[str] = Field(default_factory=list)


class IntentClassification(BaseModel):
    """Structured output from routing phase LLM call."""

    analysis_type: str = ""
    populations: list[str] = Field(default_factory=list)
    chromosomes: list[int] = Field(default_factory=list)
    focus: str = ""


class UserResponse(BaseModel):
    """User's response to a validation gate."""

    action: str  # "approve", "refine", "abort"
    feedback: str = ""


# --- Central State Object ---


class PipelineState(BaseModel):
    """Complete state of a single pipeline execution.

    The Conductor's accumulated context — serializable for Temporal
    activity boundaries and crash recovery. All fields are JSON-serializable.
    Designed for serialization from day one (Temporal compatibility).
    """

    model_config = ConfigDict(use_enum_values=False)

    # Identity
    execution_id: str = ""
    created_at: str = ""

    # Current status
    current_phase: PipelinePhase = PipelinePhase.ROUTING
    status: PipelineStatus = PipelineStatus.PENDING

    # User input
    user_prompt: str = ""

    # Conversation history (for context synthesis to stateless agents)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    # Phase outputs (accumulated across the pipeline)
    workflow_plan: WorkflowPlan | None = None
    resource_profiles: list[ResourceProfile] = Field(default_factory=list)
    infrastructure: InfrastructureMeasurements | None = None
    chromosome_data: list[ChromosomeData] = Field(default_factory=list)
    workflow_json: dict[str, Any] | None = None
    workflow_json_path: str = ""
    execution_model_recommendation: ExecutionModelRecommendation | None = None
    execution_summary: ExecutionSummary | None = None

    # Phase-specific fields
    intent_classification: str = ""
    user_approved_plan: bool = False
    user_modifications: str = ""
    namespace: str = ""
    cluster_ready: bool = False
    helm_release_name: str = ""
    engine_pod_name: str = ""
    workflow_status: str = ""
    task_completion_count: int = 0
    total_task_count: int = 0
    teardown_completed: bool = False
    user_approved_execution: bool = False

    # Data preparation outputs (for generation/deployment phases)
    vcf_header: str = ""  # #CHROM line from VCF, for columns.txt generation
    columns_txt: str = ""  # Population-filtered columns.txt content
    population_files: dict[str, str] = Field(default_factory=dict)

    # Phase v2 hooks (populated in later stages)
    resource_profile: dict[str, Any] | None = None  # From profiler (Stage 3)
    cluster_environment: dict[str, Any] | None = None  # From discovery (Stage 7)

    # Audit trail
    phase_results: list[PhaseResult] = Field(default_factory=list)
    planner_history: list[dict[str, Any]] = Field(default_factory=list)
    phase_timings: dict[str, float] = Field(default_factory=dict)

    # Execution time estimate (filled by estimation phase after executor_selection)
    estimated_duration_seconds: float = 0.0

    # Estimation accuracy (filled after monitoring completes)
    # Keys: estimated_s, actual_s, error_pct
    estimation_accuracy: dict[str, float] = Field(default_factory=dict)

    # Experiment data capture (for paper reporting)
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    planning_estimates: dict[str, Any] = Field(default_factory=dict)
    cluster_snapshots: list[dict[str, Any]] = Field(default_factory=list)

    errors: list[PipelineError] = Field(default_factory=list)

    def record_phase(self, result: PhaseResult) -> None:
        """Append a phase result to the audit trail."""
        self.phase_results.append(result)
        if result.duration_seconds > 0:
            self.phase_timings[result.phase.value] = result.duration_seconds

    def add_conversation(self, role: str, content: str) -> None:
        """Track conversation turns for context synthesis."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def synthesize_context_for_composer(self) -> str:
        """Build a single coherent context string for stateless Composer calls.

        Assembles: original query + previous plan + user feedback from
        validation gates + infrastructure measurements.
        """
        parts: list[str] = []

        parts.append(f"Original research query: {self.user_prompt}")

        if self.workflow_plan:
            parts.append(
                f"Previous plan: {self.workflow_plan.description}"
                f" (chromosomes={self.workflow_plan.chromosomes},"
                f" populations={self.workflow_plan.populations},"
                f" parallelism={self.workflow_plan.parallelism})"
            )

        if self.user_modifications:
            parts.append(f"User feedback: {self.user_modifications}")

        if self.infrastructure:
            parts.append(
                f"Infrastructure: {self.infrastructure.node_count} nodes,"
                f" {self.infrastructure.available_vcpus} vCPUs,"
                f" {self.infrastructure.memory_gb}GB RAM"
            )

        return "\n\n".join(parts)

    def to_json(self) -> str:
        """Serialize to JSON string for Temporal or persistence."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineState:
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)
