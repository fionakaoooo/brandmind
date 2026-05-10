# Design Decisions

## Why LangGraph?
- Native support for stateful multi-agent graphs
- Conditional edges for revision loop routing
- Explicit state passing between agents
- Easy to add new agents or modify flow

## Why Separate QC Agent?
- Self-Refine (Madaan 2023) uses single model as both generator and critic
- Separate QC agent reduces self-enhancement bias
- Programmatic WCAG check impossible with single-model approach
- Clearer separation of responsibilities

## Why Tool-Grounded Generation?
- LLMs hallucinate font names and color codes
- Grounding in Google Fonts ensures fonts actually exist
- Kaggle palette dataset provides human-validated colors
- Reduces hallucination while maintaining creativity

## Why Anchor Colors?
- Mid-tone palettes fail WCAG against each other
- #FFFFFF and #1A1A1A guarantee at least one compliant pair
- Real brand kits always include near-black and near-white
- Justified design decision, not a hack

## Why gpt-4o-mini?
- Strong structured JSON generation
- Cost-effective for multi-agent pipeline
- Same model across all systems for fair comparison
- Architecture-agnostic — easy to swap models
