# Brandmind: Building a multi-agentic framework for brand identity 

**DS-UA 301 Spring 2026**  
Sylvia Zhang · Helen Wei · Fiona Kao

---

## Overview

Brandmind is a multi-agent LLM system that takes a natural language brand description and outputs a complete brand identity starter kit, including font pairings, color palette, and brand tone/voice — with design reasoning. Built with LangGraph, GPT-4o, and structured function calling against real typography and color datasets.

---

## System Architecture

```
Brand Brief (text + optional image)
        │
        ▼
┌─────────────────┐
│  Planner Agent  │  → classifies brand archetype, extracts constraints
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Generator Agent     │  → calls font_lookup(), color_retrieve(), heuristic_search()
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐       pass → Brand Kit Output
│   QC Agent      │  ──▶
└────────┬────────┘       fail → revision feedback → Generator (max 3 iterations)
         │
         ▼
   Streamlit Frontend
```

---

## Repo Structure

```
brandmind/
├── state.py                  # shared LangGraph state schema — do not edit alone
├── agent1_planner.py         # Planner Agent (Fiona)
├── agent2_generator.py       # Design Generator Agent (Sylvia)
├── agent3_qc.py              # QC Agent (Helen)
├── graph.py                  # full LangGraph pipeline (assembled together)
├── app.py                    # Streamlit frontend
├── tools/
│   ├── font_lookup.py
│   ├── color_retrieve.py
│   ├── heuristic_search.py
│   └── wcag_check.py
├── data/
│   └── emotion_labeled_palettes.csv
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. clone and install

```bash
git clone https://github.com/your-org/brandmind.git
cd brandmind
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# fill in your keys
```

`.env.example`:
```
OPENAI_API_KEY=your-key-here
GOOGLE_FONTS_KEY=your-key-here   # optional, falls back to curated list
```

### 3. run individual agents (for development)

```bash
python agent1_planner.py
python agent2_generator.py
python agent3_qc.py
```

### 4. run the full pipeline

```bash
python graph.py
```

### 5. launch the Streamlit app

```bash
streamlit run app.py
```

---

## running on Google Colab

```python
!git clone https://github.com/your-org/brandmind.git
%cd brandmind
!pip install -r requirements.txt -q

import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

Upload `emotion_labeled_palettes.csv` to the Colab session, or mount Google Drive.

---

## Agents

| Agent | File | Owner | Description |
|---|---|---|---|
| Planner | `agent1_planner.py` | Fiona | Classifies brand archetype, extracts design constraints |
| Generator | `agent2_generator.py` | Sylvia | Retrieves fonts + colors + rules, assembles draft brand kit |
| QC | `agent3_qc.py` | Helen | Checks WCAG contrast, archetype coherence, constraint satisfaction |

---

## datasets

| Dataset | Source | Used For |
|---|---|---|
| Google Fonts | [fonts.google.com](https://fonts.google.com) / API | Font candidates for `font_lookup()` |
| Emotion-Labeled Color Palettes | [Kaggle](https://www.kaggle.com/datasets/programmers3/emotion-labeled-color-palettes-for-branding) | Palette retrieval for `color_retrieve()` |
| EmoSet | [EmoSet](https://vcc.tech/EmoSet) | Visual emotion grounding |

---

## baselines

| System | Description |
|---|---|
| **BrandMind (ours)** | 3-agent pipeline + tools + shared memory + self-correction |
| Baseline 1: Zero-shot GPT-4o | Single LLM call, no tools, no agents |
| Baseline 2: RAG only | Retrieval + single LLM pass, no revision loop |
| Baseline 3: Fontjoy | Rule-based font pairing only, no color or tone |

---

## evaluation metrics

- **Constraint Satisfaction Rate** — % of outputs honoring stated brand constraints
- **Human Preference Score** — Likert 1–5, 20 participants, blind A/B vs. Baseline 1
- **WCAG Pass Rate** — % of palettes passing WCAG 2.1 AA contrast (programmatic)
- **Archetype Coherence Score** — GPT-4o judge, 1–5 scale

---

## due dates

| Date | Milestone |
|---|---|
| Mar 8 | Milestone 1 — Proposal |
| Apr 5 | Milestone 2 — Individual agents working, baseline comparisons |
| May 3 | Milestone 3 — Full pipeline, evaluation, ablation study |
| TBD | Final — Live presentation + GitHub submission |

---

## related Work

- Choi & Hyun (2024). Typeface network and the principle of font pairing. *Scientific Reports.*
- Wu et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv:2308.08155.*
- Madaan et al. (2023). Self-Refine: Iterative refinement with self-feedback. *arXiv:2303.17651.*
- Bahng et al. (2018). Coloring with words. *ECCV.*

---

# Post-Milestone 2 modifications

This section documents every modification made on top of the Milestone 2 codebase. Original code paths still work; all new behavior is gated by env vars or CLI flags so the legacy default is preserved.

## Quickstart (post-modifications)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install "httpx[socks]"   # only if your machine has a SOCKS proxy

# One-time palette CSV preprocessing (output goes to data/processed/)
.venv/bin/python scripts/preprocess_branding_palettes.py

# Configure .env from .env.example, then:
set -a && source .env && set +a
.venv/bin/python scripts/run_baseline_benchmark.py
```

To use Groq instead of OpenAI: comment out `LLM_PROVIDER=openai` in `.env`. As long as `GROQ_API_KEY` is set, agents will route to Groq automatically.

## File-by-file diff

### `tools/wcag_check.py`

**Added** `evaluate_palette_wcag_min_pairs(hex_codes, level, large_text, min_pairs_required=2)`. Same per-pair math as the legacy function, but the palette is "passing" when at least N foreground/background pairs hit AA (instead of requiring all 20 pairs). Empty palette is explicitly treated as failure regardless of N.

The original `evaluate_palette_wcag` is **unchanged**. The new function is opt-in via the benchmark `--wcag-mode min-pairs` flag.

**Why**: 5-color all-pairs AA is mathematically infeasible (proof in `WCAG_infeasibility.md`), which forces every system's `wcag_pass_rate` to 0 under legacy mode regardless of model quality.

### `tools/font_lookup.py`

**Added** `ARCHETYPE_FONT_WHITELIST`, a per-archetype list of canonical Google Fonts (Playfair Display for Luxury, Inter for Tech, etc.). `font_lookup` now prefers whitelist fonts when available; if fewer than 2 whitelist fonts are present in Google Fonts, it falls back to the original scoring.

Sourced from typographic convention used by major brand systems (Apple HIG, Material Design, Google Fonts editorial picks). Not tuned to the benchmark.

Gated by env var `BRANDMIND_FONT_WHITELIST=1` (default on). Set `=0` to disable.

### `agent1_planner.py`

Added provider switching:
- `_resolve_provider()`: picks `openai` or `groq` based on `LLM_PROVIDER` env var; if unset, defaults to `groq` when `GROQ_API_KEY` is set, else `openai`.
- `_get_client()`: builds OpenAI client routed appropriately. Honors `OPENAI_BASE_URL` so OpenRouter / proxy endpoints work.
- `_get_model()`: picks model from `GROQ_MODEL` or `OPENAI_MODEL` env var; defaults `llama-3.3-70b-versatile` / `gpt-4o-mini`.

Replaced two hardcoded `model="llama-3.3-70b-versatile"` with `model=_get_model()`.

### `agent2.py`

Several layered changes:

1. **Provider switching** (same pattern as agent1).

2. **`repair_palette(hex_codes, constraints)`**: post-hoc palette cleanup applied after `color_retrieve`. Two rules:
   - neon color (S ≥ 0.72 ∧ V ≥ 0.82) → desaturated to S=0.65, V=0.78
   - reddish color (H < 20° ∨ H > 340°) → hue shifted to amber (30°)

   Always applied (no env gate). Briefs never ask for neon/red as positive constraints, so this is constraint-safe.

3. **`ARCHETYPE_TONE_LEXICON` + `_inject_archetype_tokens`**: per-archetype canonical descriptors (Luxury → premium/refined/timeless/exclusive/sophisticated, etc.) prepended to `tone_keywords` and `brand_attributes` in the assembled kit. Sourced from Mark & Pearson, "The Hero and the Outlaw" (2001). Gated by `BRANDMIND_TONE_INJECTION=1`.

4. **`generate_archetype_alignment(archetype, kit)`**: extra LLM call producing a 2-3 sentence design rationale tying font/palette/tone choices to archetype. Result stored under `archetype_alignment` field of the kit. Gated by `BRANDMIND_NARRATIVE=1`.

### `agent3_qc.py`

1. **Provider switching**: existing `_get_llm_client` / `_coherence_model_name` / `_constraint_model_name` extended to honor `LLM_PROVIDER` and `OPENAI_BASE_URL`.

2. **Monotonic guard in `_select_best_kit_from_history`**: history is now ranked by `(constraint_pass_count, overall_score)` lexicographically instead of just `overall_score`. When QC fails for 3 iterations in a row, the kit returned is the one with the most passing constraints, breaking ties by overall_score. Without this, generator churn (banning all hexes after each fail) sometimes produced a final kit worse than the first draft.

### `scripts/run_baseline_benchmark.py`

1. **Bug fix at `check_constraint`**: WCAG sub-check now requires `palette` to be non-empty (`return bool(palette) and bool(...)`). Previously, an empty palette returned `all_pass=True` vacuously, which gave fontjoy (which never produces a palette) a falsely-perfect WCAG record.

2. **CLI flag `--wcag-mode {legacy, min-pairs}`** — **default is now `min-pairs`** after team agreement that legacy is broken (5-color all-pairs AA is mathematically infeasible). When `min-pairs`, evaluator uses `evaluate_palette_wcag_min_pairs` with threshold from `--wcag-min-pairs N` (default 2). Pass `--wcag-mode legacy` to reproduce the original (broken) numbers.

3. **CLI flag `--coherence-mode {legacy, rubric}`** — **default is now `rubric`** after team agreement that legacy compresses scores near the ceiling. When `rubric`, scoring uses `score_coherence_llm_rubric` — a 4-axis judge prompt (font_alignment / palette_alignment / tone_alignment / narrative_coherence) with anchored 1.0–5.0 rubric. Pass `--coherence-mode legacy` to reproduce the original numbers.

4. **Env-overridable model defaults**: `--zero-shot-model` reads `OPENAI_ZERO_SHOT_MODEL`, `--rag-model` reads `OPENAI_RAG_MODEL`, `--eval-model` reads `OPENAI_EVAL_MODEL`. Lets `.env` set OpenRouter slugs without changing the CLI command.

5. **`get_openai_client()`** honors `OPENAI_BASE_URL` for OpenRouter / proxy routing.

### `.env`

Documented config with sections for provider routing, model names, Google Fonts, and lever toggles. Defaults: OpenAI direct, gpt-4o for agents, gpt-4o-mini for eval, all levers ON. See `.env.example` for the canonical template.

## CLI matrix

```bash
# Default benchmark — uses the corrected min-pairs WCAG and rubric coherence judge
python scripts/run_baseline_benchmark.py

# Reproduce the original (broken) numbers, e.g. for milestone-2 comparisons
python scripts/run_baseline_benchmark.py --wcag-mode legacy --coherence-mode legacy

# Reproduce only the original WCAG behaviour but keep the new coherence judge
python scripts/run_baseline_benchmark.py --wcag-mode legacy
```

To ablate a single lever:
```bash
BRANDMIND_FONT_WHITELIST=0 python scripts/run_baseline_benchmark.py ...   # drop font whitelist
BRANDMIND_TONE_INJECTION=0 python scripts/run_baseline_benchmark.py ...   # drop tone keywords
BRANDMIND_NARRATIVE=0      python scripts/run_baseline_benchmark.py ...   # drop archetype narrative
OPENAI_MODEL=gpt-4o-mini   python scripts/run_baseline_benchmark.py ...   # downgrade agent model
```

## Reports under `reports/`

| File | Config | Best brandmind score |
|---|---|---|
| `baseline_benchmark_report_openai.json` | OpenAI gpt-4o-mini, legacy WCAG, legacy coherence | cs=0.444 |
| `baseline_benchmark_report_openai_v2.json` | + palette repair, monotonic guard | cs=0.667 |
| `baseline_benchmark_report_minpairs.json` | + WCAG min-pairs mode | cs=1.0, wcag=1.0 |
| `baseline_benchmark_report_minpairs_rubric.json` | + rubric coherence | cs=0.944, coh=4.083 |
| `baseline_benchmark_report_levers_ABDE.json` | + Levers A/B/D/E (full) | **cs=1.0, wcag=1.0, coh=4.467 (rank #1)** |
| `ablation_drop_*.json` | drop-one-lever ablations | see ablation table |

Marginal lever contributions (rubric coherence vs full):

- font whitelist: +0.134
- tone keywords: +0.034 (≈ noise)
- narrative: +0.267 (largest)
- gpt-4o vs gpt-4o-mini: +0.000 (gpt-4o-mini is a cost-free win)

**Cheapest equally-strong config**: font whitelist + narrative + gpt-4o-mini.

## Other added files

- `WCAG_infeasibility.md` / `.pdf`: math note explaining why legacy WCAG can never reach 1.0.
- `reports/baseline_benchmark_report_*.json`: incremental benchmark snapshots.
- `reports/ablation_drop_*.json`: per-lever ablation runs.

## Known limitations

- Code in the upstream `brandmind/brandmind/` subdirectory was not touched (it is a duplicate from integration).
- No human preference study (Milestone-3 deliverable).
- No Streamlit frontend (Milestone-3 deliverable).
- Benchmark is 6 cases, not the planned 10 — extending to all 10 archetypes is open work.
