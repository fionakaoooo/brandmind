# Model Selection Rationale

## Chosen Model: gpt-4o-mini (OpenAI)

### Why gpt-4o-mini
- Strong performance on structured JSON generation
- Reliable response_format=json_object support
- Cost-effective for multi-agent pipeline with 3+ LLM calls per run
- Fast inference suitable for iterative revision loop

### Experimental Control
All systems (BrandMind, Zero-shot, RAG-only) use the same model.
This isolates architectural differences rather than model capability.
Any performance gap is attributable to the pipeline design.

### Previous: Groq llama-3.3-70b-versatile
- Used during development for free tier access
- Switched to gpt-4o-mini for final evaluation
- Architecture is model-agnostic (2-line change to switch)
