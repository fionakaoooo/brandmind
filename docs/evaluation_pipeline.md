# Evaluation Pipeline

## Step 1: Baseline Benchmark
Run all 4 systems on 20 brand briefs:
```bash
python3 scripts/run_baseline_benchmark.py \
  --out reports/results.json \
  --wcag-min-pairs 1
```

## Step 2: Analysis Scripts
```bash
python3 analysis/archetype_analysis.py
python3 analysis/wcag_analysis.py
python3 analysis/csr_analysis.py
python3 analysis/coherence_analysis.py
```

## Step 3: Generate Charts
```bash
python3 analysis/per_archetype_chart.py
python3 analysis/human_study_charts.py
```

## Step 4: Human Study
- Collect responses via Google Form
- Download CSV to reports/
- Run: python3 analysis/human_study_statistics.py
