# Agent Interaction Pattern

## Communication via Shared State
Agents do not call each other directly.
All communication happens through BrandMindState TypedDict.

## Handoff: Agent 1 → Agent 2
Agent 1 writes:
- archetype, archetype_rationale
- design_constraints, constraints
- status = "generating"

Agent 2 reads:
- archetype (to select fonts/colors)
- constraints (to guide generation)

## Handoff: Agent 2 → Agent 3
Agent 2 writes:
- draft_brand_kit
- revision_history
- iteration_count
- status = "reviewing"

Agent 3 reads:
- draft_brand_kit (to evaluate)
- design_constraints (to verify)
- iteration_count (to check max)

## Handoff: Agent 3 → Agent 2 (revision)
Agent 3 writes:
- qc_feedback (specific revision instructions)
- qc_scores (WCAG, coherence, CSR)
- status = "generating" (triggers revision)
