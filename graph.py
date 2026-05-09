"""
graph.py — BrandMind LangGraph pipeline

Wires Agent 1 (Planner), Agent 2 (Generator), and Agent 3 (QC)
into a StateGraph with conditional edges, replacing the hand-written
run_pipeline() loop.

Usage:
    from graph import build_graph

    graph = build_graph()
    result = graph.invoke({"brand_brief": "...", ...})
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END

from state import BrandMindState
from agent1_planner import planner_agent
from agent2 import design_generator_agent
from agent3_qc import qc_agent


# ─────────────────────────────────────────────────────────────────────────────
# Node wrappers
# Each wrapper just calls the agent function and returns the updated state.
# LangGraph merges the returned dict back into the shared state automatically.
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: BrandMindState) -> BrandMindState:
    """Agent 1: classify archetype, extract constraints, initialise weights."""
    print("\n[Graph] → planner_node")
    return planner_agent(state)


def generator_node(state: BrandMindState) -> BrandMindState:
    """Agent 2: retrieve fonts/colors/heuristics, assemble draft brand kit."""
    print("\n[Graph] → generator_node")
    return design_generator_agent(state)


def qc_node(state: BrandMindState) -> BrandMindState:
    """Agent 3: WCAG check, coherence score, constraint verification."""
    print("\n[Graph] → qc_node")
    return qc_agent(state)


# ─────────────────────────────────────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 3


def route_after_qc(
    state: BrandMindState,
) -> Literal["generator", "__end__"]:
    """
    Conditional edge evaluated after every QC pass.

    Returns:
        "generator"  — QC failed and we still have iterations left → revise
        "__end__"    — approved OR max iterations reached → terminate
    """
    status = state.get("status", "")
    iteration = int(state.get("iteration_count", 0))

    if status == "approved":
        print(f"[Graph] QC approved. Terminating after {iteration} iteration(s).")
        return END

    if status == "failed" or iteration >= MAX_ITERATIONS:
        print(f"[Graph] Max iterations ({MAX_ITERATIONS}) reached. Terminating.")
        return END

    print(f"[Graph] QC failed (iteration {iteration}). Routing back to generator.")
    return "generator"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Build and compile the BrandMind StateGraph.

    Graph structure:
        START → planner → generator → qc ─┐
                                   ↑      │  (approved / failed → END)
                                   └──────┘  (retry → generator)

    Returns:
        A compiled LangGraph runnable (supports .invoke(), .stream()).
    """
    graph = StateGraph(BrandMindState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("generator", generator_node)
    graph.add_node("qc", qc_node)

    # Entry point
    graph.set_entry_point("planner")

    # Planner always flows to generator
    graph.add_edge("planner", "generator")

    # Generator always flows to QC
    graph.add_edge("generator", "qc")

    # QC uses conditional edge: retry generator OR terminate
    graph.add_conditional_edges(
        "qc",
        route_after_qc,
        {
            "generator": "generator",
            END: END,
        },
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner (mirrors the old run_pipeline() signature)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(brand_brief: str, max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
    """
    Drop-in replacement for the hand-written run_pipeline().

    Args:
        brand_brief:     Natural language brand description.
        max_iterations:  Upper bound on generator→QC cycles (default 3).

    Returns:
        Final BrandMindState dict with approved_brand_kit (or best draft).
    """
    initial_state: BrandMindState = {
        "brand_brief": brand_brief,
        "clip_features": None,
        "archetype": None,
        "archetype_rationale": None,
        "design_constraints": None,
        "constraints": None,
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

    compiled = build_graph()
    final_state = compiled.invoke(initial_state)

    status = final_state.get("status")
    iterations = final_state.get("iteration_count", 0)
    print(f"\n[Pipeline] Done. status={status}, iterations={iterations}")

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# Manual smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TEST_BRIEF = (
        "We are launching a sustainable skincare brand called Verdant targeting "
        "eco-conscious women aged 25-40. The brand should feel premium but warm "
        "and grounded in nature. The color palette must be WCAG AA accessible. "
        "No neon colors."
    )

    result = run_pipeline(TEST_BRIEF)

    print("\n=== Final state keys ===")
    for k, v in result.items():
        if k == "approved_brand_kit":
            print(f"  {k}: {json.dumps(v, indent=2, ensure_ascii=False)[:400]}...")
        elif k == "heuristic_weights":
            non_default = {rid: w for rid, w in (v or {}).items() if w != 1.0}
            print(f"  {k}: {len(v or {})} rules, {len(non_default)} updated")
        else:
            print(f"  {k}: {v}")
