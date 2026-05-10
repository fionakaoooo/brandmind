# Brand Brief Validation

Agent 1 validates the brand brief before running the pipeline.

## Checks
- Word count >= 15 (sufficient detail)
- Word count <= 500 (not too verbose)
- Contains brand/company reference
- Contains target audience reference

## Warnings
If validation fails, warnings are printed but pipeline continues.
Future work: reject briefs below quality threshold.
