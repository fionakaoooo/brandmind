# Benchmark Methodology

## Test Cases
- 20 brand briefs spanning 10 archetypes
- Each brief has 3-4 explicit constraints
- Cases cover diverse industries

## Systems Evaluated
1. BrandMind (full pipeline)
2. Zero-shot GPT-4o-mini
3. RAG-only GPT-4o-mini
4. Fontjoy (rule-based)

## Evaluation Metrics
- CSR: fraction of constraints passing
- WCAG: fraction of cases with >= 1 compliant color pair
- Coherence: LLM judge 1-5 rubric score

## Experimental Controls
- Same LLM (gpt-4o-mini) for all systems
- Same evaluation prompt for all systems
- Same 20 briefs for all systems
- Randomized kit assignment in human study
