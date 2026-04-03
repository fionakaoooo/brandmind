from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agent1_planner import planner_agent
from agent2 import design_generator_agent
from agent3_qc import MAX_ITERATIONS, qc_agent
from state import BrandMindState


DEFAULT_BRIEF = (
    "Aurora Trail is an eco-friendly outdoor apparel brand for urban hikers aged 22-40. "
    "The identity should feel trustworthy, modern, and grounded in nature. "
    "Avoid neon colors and keep the palette WCAG-friendly for text overlays. "
    "Typography should feel clean and premium but not cold."
)


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _to_generator_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    out["constraints"] = state.get("design_constraints", []) or state.get("constraints", []) or []
    out["clip_context"] = ""
    return out


def _initial_state(brief: str, clip_features: Optional[list] = None) -> BrandMindState:
    return {
        "brand_brief": brief,
        "clip_features": clip_features,
        "archetype": None,
        "archetype_rationale": None,
        "design_constraints": None,
        "design_spec": None,
        "generator_trace": None,
        "draft_brand_kit": None,
        "qc_feedback": None,
        "qc_scores": None,
        "iteration_count": 0,
        "status": "planning",
        "revision_history": [],
        "approved_brand_kit": None,
    }


def run_pipeline(
    brand_brief: str,
    clip_features: Optional[list] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> BrandMindState:
    state = _initial_state(brand_brief, clip_features=clip_features)
    state = planner_agent(state)

    hard_guard = max(1, int(max_iterations)) + 2
    steps = 0

    while steps < hard_guard:
        steps += 1
        state = design_generator_agent(_to_generator_state(state))
        state = qc_agent(state)
        status = state.get("status")
        if status in {"approved", "failed"}:
            break
        if status != "generating":
            break

    return state


def summarize_output(state: Dict[str, Any]) -> Dict[str, Any]:
    kit = state.get("approved_brand_kit") or state.get("draft_brand_kit") or {}
    qc_scores = state.get("qc_scores", {}) if isinstance(state.get("qc_scores"), dict) else {}
    wcag = qc_scores.get("wcag", {}) if isinstance(qc_scores.get("wcag"), dict) else {}
    coherence = qc_scores.get("coherence", {}) if isinstance(qc_scores.get("coherence"), dict) else {}
    constraints = qc_scores.get("constraints", {}) if isinstance(qc_scores.get("constraints"), dict) else {}

    return {
        "status": state.get("status"),
        "iteration_count": state.get("iteration_count"),
        "archetype": state.get("archetype"),
        "constraint_count": len(state.get("design_constraints") or []),
        "approved": qc_scores.get("approved"),
        "overall_score": qc_scores.get("overall_score"),
        "wcag_pass_rate": wcag.get("pass_rate"),
        "coherence_score": coherence.get("score"),
        "constraint_pass_rate": constraints.get("pass_rate"),
        "headline_font": (
            (kit.get("font_recommendation", {}) or {})
            .get("headline", {})
            .get("family")
        ),
        "body_font": (
            (kit.get("font_recommendation", {}) or {})
            .get("body", {})
            .get("family")
        ),
        "palette_hex_codes": (kit.get("color_palette", {}) or {}).get("hex_codes", []),
        "qc_feedback": state.get("qc_feedback"),
        "revision_history_len": len(state.get("revision_history") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full BrandMind pipeline.")
    parser.add_argument("--brief", type=str, default=DEFAULT_BRIEF)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    load_dotenv_file(Path(".env"))

    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Neither GROQ_API_KEY nor OPENAI_API_KEY is set.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    try:
        final_state = run_pipeline(
            brand_brief=args.brief,
            clip_features=None,
            max_iterations=args.max_iterations,
        )
        payload = {
            "ok": True,
            "summary": summarize_output(final_state),
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"runtime_exception: {exc}",
        }

    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    print(payload_json)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload_json + "\n", encoding="utf-8")

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
