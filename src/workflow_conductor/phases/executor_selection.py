"""Phase 6b: Executor Selection — choose HyperFlow execution model.

Analyzes the generated workflow.json and selects the optimal execution
model (JOB / JOB_AGGLOMERATION / WORKER_POOL) based on workflow metrics
and available cluster vCPUs. Writes workflow.config.json to the engine pod
during deployment if a non-default model is selected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from execution_model_advisor import Advisor

from workflow_conductor.models import PipelinePhase, PipelineState
from workflow_conductor.ui.display import display_phase_header

if TYPE_CHECKING:
    from workflow_conductor.config import ConductorSettings

logger = logging.getLogger(__name__)


async def run_executor_selection_phase(
    state: PipelineState,
    settings: ConductorSettings,
) -> PipelineState:
    """Select execution model based on workflow.json metrics."""
    display_phase_header(PipelinePhase.EXECUTOR_SELECTION)

    if not state.workflow_json:
        logger.warning("No workflow.json available — skipping executor selection, defaulting to JOB")
        return state

    vcpus = state.infrastructure.available_vcpus if state.infrastructure else 0
    recommendation = Advisor.analyze(state.workflow_json, available_vcpus=vcpus)
    state.execution_model_recommendation = recommendation

    logger.info(
        "Execution model: %s (confidence=%.2f) — %s",
        recommendation.model,
        recommendation.confidence,
        recommendation.reasoning,
    )
    if recommendation.hyperflow_config:
        logger.info("workflow.config.json will be deployed: %s", recommendation.hyperflow_config)

    return state
