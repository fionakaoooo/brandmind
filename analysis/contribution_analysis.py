"""
Regenerate ablation study summary and chart from per-variant benchmark reports.

Reads:
  reports/baseline_benchmark_report_20cases_levers.json   (full ABC config)
  reports/ablation_drop_A.json
  reports/ablation_drop_B.json
  reports/ablation_drop_C.json

Writes:
  reports/ablation_summary.json
  reports/ablation_chart.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

ABLATION_VARIANTS = [
    ("drop_A", "ablation_drop_A.json"),
    ("drop_B", "ablation_drop_B.json"),
    ("drop_C", "ablation_drop_C.json"),
]


def _summary_for(report_path: Path) -> dict:
    payload = json.loads(report_path.read_text())
    bm = payload["summary"]["brandmind_full"]
    return {
        "csr": round(bm["constraint_satisfaction_rate"], 3),
        "wcag": round(bm["wcag_pass_rate"], 3),
        "coherence": round(bm["archetype_coherence_score_mean"], 3),
    }


def build_summary() -> dict:
    out = {}
    for label, fname in ABLATION_VARIANTS:
        path = REPORTS_DIR / fname
        if not path.exists():
            continue
        out[label] = _summary_for(path)
    return out


def write_summary(summary: dict) -> Path:
    out_path = REPORTS_DIR / "ablation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return out_path


def render_chart(summary: dict) -> Path:
    labels = list(summary.keys())
    csr = [summary[k]["csr"] for k in labels]
    wcag = [summary[k]["wcag"] for k in labels]
    coh_scaled = [summary[k]["coherence"] / 5.0 for k in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    x = list(range(len(labels)))

    ax.bar([p - width for p in x], csr, width, label="CSR", color="#4C9CD8")
    ax.bar(x, wcag, width, label="WCAG", color="#5FBFA0")
    ax.bar([p + width for p in x], coh_scaled, width, label="Coherence (scaled)", color="#E8924A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Ablation Variant")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Ablation Study Results")
    ax.legend(loc="upper left")

    out_path = REPORTS_DIR / "ablation_chart.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    summary = build_summary()
    if not summary:
        raise SystemExit("No ablation reports found; run drop_A/B/C benchmarks first.")
    summary_path = write_summary(summary)
    chart_path = render_chart(summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {chart_path}")
    for variant, scores in summary.items():
        print(f"  {variant}: csr={scores['csr']}  wcag={scores['wcag']}  coh={scores['coherence']}")


if __name__ == "__main__":
    main()
