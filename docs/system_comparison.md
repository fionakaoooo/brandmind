# System Comparison

## BrandMind vs Baselines

### BrandMind (ours)
- 3-agent pipeline: Planner → Generator → QC
- Tool-grounded retrieval (Google Fonts, Kaggle palettes)
- Shared LangGraph state
- Iterative self-correction (max 3 iterations)
- Programmatic WCAG checking

### Baseline 1: Zero-shot
- Single LLM call
- No tools, no memory, no revision
- Fastest but least grounded

### Baseline 2: RAG only
- Retrieval + single LLM pass
- No agents, no revision loop
- Better grounding than zero-shot but no verification

### Baseline 3: Fontjoy
- Rule-based font pairing only
- No color output, no tone, no brand context
- Cannot satisfy any color or tone constraints
