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
├── agent2.py                 # Design Generator Agent (Sylvia)
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

See `.env.example` in the repo for the full template. The required keys are
`OPENAI_API_KEY` (or `GROQ_API_KEY`), `GOOGLE_FONTS_API_KEY`, and the lever
toggles `BRANDMIND_FONT_WHITELIST` / `BRANDMIND_TONE_INJECTION` /
`BRANDMIND_NARRATIVE` (all default on). `LLM_PROVIDER=openai` selects OpenAI
direct; comment it out to auto-route to Groq when `GROQ_API_KEY` is set.

### 3. run individual agents (for development)

```bash
python agent1_planner.py
python agent2.py
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
| Generator | `agent2.py` | Sylvia | Retrieves fonts + colors + rules, assembles draft brand kit |
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

## Reports under `reports/`

| File | Config | Best brandmind score |
|---|---|---|
| `baseline_benchmark_report_openai.json` | OpenAI gpt-4o-mini, legacy WCAG, legacy coherence (6 cases) | cs=0.444 |
| `baseline_benchmark_report_openai_v2.json` | + palette repair, monotonic guard (6 cases) | cs=0.667 |
| `baseline_benchmark_report_minpairs.json` | + WCAG min-pairs (6 cases) | cs=1.0, wcag=1.0 |
| `baseline_benchmark_report_minpairs_rubric.json` | + rubric coherence (6 cases) | cs=0.944, coh=4.083 |
| `baseline_benchmark_report_levers_ABC.json` | + Levers A/B/C (full, 20 cases) | **cs=0.716, wcag=0.950, coh=4.456 (rank #2)** |
| `ablation_drop_*.json` | drop-one-lever ablations (20 cases) | see ablation table |

Marginal lever contributions on rubric coherence (full ABC vs drop-one):

- A — archetype font whitelist: **+0.196**
- B — archetype tone keywords: +0.041 (≈ noise)
- C — archetype alignment narrative: **+0.151**

Run a custom configuration:

```bash
# default - all levers ABC on, gpt-4o-mini, min-pairs WCAG, rubric coherence
python scripts/run_baseline_benchmark.py

# ablation: drop one lever via env var
BRANDMIND_FONT_WHITELIST=0 python scripts/run_baseline_benchmark.py   # drop A
BRANDMIND_TONE_INJECTION=0 python scripts/run_baseline_benchmark.py   # drop B
BRANDMIND_NARRATIVE=0      python scripts/run_baseline_benchmark.py   # drop C

# reproduce the original (broken) numbers
python scripts/run_baseline_benchmark.py --wcag-mode legacy --coherence-mode legacy
```
