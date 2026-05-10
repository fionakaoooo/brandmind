# Known Limitations

## Color Retrieval
- Kaggle palette dataset has limited diversity
- Mid-tone palettes dominate — insufficient contrast range
- Cannot always satisfy all constraints simultaneously

## QC Agent
- 0/20 cases approved in 20-case benchmark
- Strict thresholds prevent approval even for good kits
- Rule-based checker over-penalizes borderline colors

## Runtime
- 37s average vs 4s for zero-shot
- All 3 iterations always used (never early termination)
- Not suitable for real-time applications

## Human Study
- 20 participants — small sample size
- Informal recruitment — possible selection bias
- Only 5 of 20 briefs evaluated

## Evaluation
- LLM judge uses same model as generator (self-enhancement bias)
- CSR metric stricter than human judgment
- No evaluation on real startup use cases
