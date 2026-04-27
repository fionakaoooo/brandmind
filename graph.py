from __future__ import annotations
from typing import Any, Dict
from state import BrandMindState
from agent1_planner import planner_agent
from agent2 import design_generator_agent
from agent3_qc import qc_agent


def run_pipeline(brand_brief: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    Run the full BrandMind pipeline.
    Returns the final state dict.
    """
    state: BrandMindState = {
        "brand_brief": brand_brief,
        "clip_features": None,
        "archetype": None,
        "archetype_rationale": None,
        "design_constraints": None,
        "constraints": None,          # 修改2：新增
        "design_spec": None,
        "generator_trace": None,
        "draft_brand_kit": None,
        "qc_feedback": None,
        "qc_scores": None,
        "heuristic_weights": None,
        "iteration_count": 0,
        "status": "planning",
        "revision_history": [],
        "approved_brand_kit": None,
    }

    # Agent 1: Plan
    state = planner_agent(state)

    # Agent 2 + 3: Generate → QC loop
    for i in range(max_iterations):
        state = design_generator_agent(state)
        state = qc_agent(state)

        if state["status"] == "approved":
            print(f"[Pipeline] Approved after {i + 1} iteration(s).")
            break

        if state["status"] == "failed":
            print(f"[Pipeline] Max iterations reached. Returning best draft.")
            break

    return state
