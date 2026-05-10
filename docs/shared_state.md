# Shared State (BrandMindState)

LangGraph TypedDict passed between all agents.

## Fields
| Field | Type | Set By | Read By |
|---|---|---|---|
| brand_brief | str | User | Agent 1 |
| archetype | str | Agent 1 | Agent 2, 3 |
| design_constraints | list | Agent 1 | Agent 2, 3 |
| constraints | list | Agent 1 | Agent 2 |
| draft_brand_kit | dict | Agent 2 | Agent 3 |
| qc_scores | dict | Agent 3 | Graph router |
| qc_feedback | str | Agent 3 | Agent 2 |
| iteration_count | int | Graph | Agent 2, 3 |
| status | str | Agent 3 | Graph router |
| revision_history | list | Agent 2 | Agent 3 |
| approved_brand_kit | dict | Agent 3 | Frontend |
