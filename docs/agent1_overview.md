# Agent 1: Planner Agent

## Responsibilities
- Classifies brand brief into one of 10 archetypes
- Extracts explicit and implicit design constraints
- Detects industry from brief keywords
- Extracts target audience age range
- Writes all outputs to shared LangGraph state

## Tools Used
- archetype_classifier() — LLM-powered classification
- extract_constraints() — LLM-powered constraint extraction

## Output Fields
- archetype: one of 10 brand archetypes
- archetype_rationale: explanation of classification
- design_constraints: list of actionable constraints
- constraints: same list (for agent2 compatibility)
- status: set to "generating" to trigger Agent 2
