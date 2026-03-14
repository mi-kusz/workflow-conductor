"""Phase 8: Deployment — deliver workflow.json and signal engine to start.

After provisioning (hf-ops + hf-run), data preparation, and generation
(workflow.json), deployment:
- Writes workflow.json to a temp file
- Copies it to the engine pod via kubectl cp
- Signals the engine to start by touching .conductor-ready
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING

from workflow_conductor.k8s import Kubectl
from workflow_conductor.models import PipelinePhase, PipelineState
from workflow_conductor.ui.display import display_phase_header

if TYPE_CHECKING:
    from workflow_conductor.config import ConductorSettings

logger = logging.getLogger(__name__)


async def run_deployment_phase(
    state: PipelineState,
    settings: ConductorSettings,
) -> PipelineState:
    """Deliver workflow.json and signal the engine to start.

    Requires state.namespace and state.cluster_ready from provisioning.
    Requires state.engine_pod_name from provisioning.
    Requires state.workflow_json from generation.
    """
    display_phase_header(PipelinePhase.DEPLOYMENT)

    if not state.namespace:
        raise ValueError(
            "Cannot deploy: namespace not set (provisioning phase required)"
        )
    if not state.cluster_ready:
        raise ValueError(
            "Cannot deploy: cluster not ready (provisioning phase required)"
        )
    if not state.engine_pod_name:
        raise ValueError(
            "Cannot deploy: engine pod not set (provisioning phase required)"
        )

    namespace = state.namespace
    kubectl = Kubectl(kubeconfig=settings.kubernetes.kubeconfig)

    # Step 1: Write workflow.json to temp file and copy to engine pod
    if state.workflow_json:
        logger.info("Step 1: Delivering workflow.json to engine pod")
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(state.workflow_json, f)
            state.workflow_json_path = f.name

        await kubectl.cp_to_pod(
            state.workflow_json_path,
            state.engine_pod_name,
            "/work_dir/workflow.json",
            namespace=namespace,
            container="hyperflow",
        )
        os.unlink(state.workflow_json_path)

    # Step 2: Write columns.txt (population-filtered)
    if state.columns_txt:
        logger.info("Step 2: Deploying columns.txt to engine pod")
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write(state.columns_txt)
            columns_path = f.name
        await kubectl.cp_to_pod(
            columns_path,
            state.engine_pod_name,
            "/work_dir/columns.txt",
            namespace=namespace,
            container="hyperflow",
        )
        os.unlink(columns_path)

    # Step 3: Write population files
    if state.population_files:
        logger.info(
            "Step 3: Deploying %d population files", len(state.population_files)
        )
        for pop_name, pop_content in state.population_files.items():
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
            ) as f:
                f.write(pop_content)
                pop_path = f.name
            await kubectl.cp_to_pod(
                pop_path,
                state.engine_pod_name,
                f"/work_dir/{pop_name}",
                namespace=namespace,
                container="hyperflow",
            )
            os.unlink(pop_path)
            logger.info("  Deployed %s", pop_name)

    # Step 4: Write hyperflow.json if execution model requires it
    if (
        state.execution_model_recommendation
        and state.execution_model_recommendation.hyperflow_config
    ):
        hf_config = state.execution_model_recommendation.hyperflow_config
        logger.info("Step 4: Deploying workflow.config.json (model=%s)", state.execution_model_recommendation.model)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(hf_config, f)
            hf_path = f.name
        await kubectl.cp_to_pod(
            hf_path,
            state.engine_pod_name,
            "/work_dir/workflow.config.json",
            namespace=namespace,
            container="hyperflow",
        )
        os.unlink(hf_path)

    # Step 5: Signal engine to start
    logger.info("Step 5: Signaling engine to start")
    await kubectl.exec_in_pod(
        state.engine_pod_name,
        ["touch", "/work_dir/.conductor-ready"],
        namespace=namespace,
        container="hyperflow",
    )

    state.helm_release_name = "hf-run"
    logger.info("Deployment complete — engine signaled to start")

    return state
