"""
Agent 3: Quality Control Agent
- Reviews draft brand kit from Agent 2
- Checks WCAG accessibility for palette
- Scores archetype coherence with LLM judge
- Verifies user constraints
- Approves or sends revision feedback (max 3 iterations)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from state import BrandMindState
from tools.wcag_check import evaluate_palette_wcag


MAX_ITERATIONS = 3
COHERENCE_PASS_THRESHOLD = 3.5


def _safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _get_llm_client() -> Optional[OpenAI]:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    if openai_key:
        return OpenAI(api_key=openai_key)
    return None


def _coherence_model_name() -> str:
    if os.environ.get("GROQ_API_KEY", "").strip():
        return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _constraint_model_name() -> str:
    if os.environ.get("GROQ_API_KEY", "").strip():
        return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _extract_constraints(state: Dict[str, Any], kit: Dict[str, Any]) -> List[str]:
    merged: List[str] = []
    for source in (
        state.get("design_constraints"),
        state.get("constraints"),
        kit.get("constraints_checked_later"),
    ):
        if isinstance(source, list):
            for item in source:
                text = str(item).strip()
                if text and text not in merged:
                    merged.append(text)
    return merged


def _extract_palette(kit: Dict[str, Any]) -> List[str]:
    palette = kit.get("color_palette", {})
    if not isinstance(palette, dict):
        return []
    hex_codes = palette.get("hex_codes", [])
    if not isinstance(hex_codes, list):
        return []
    return [str(h).strip() for h in hex_codes if str(h).strip()]


def _extract_font_categories(kit: Dict[str, Any]) -> Dict[str, str]:
    fonts = kit.get("font_recommendation", {})
    if not isinstance(fonts, dict):
        return {"headline": "", "body": ""}
    headline = fonts.get("headline", {}) if isinstance(fonts.get("headline", {}), dict) else {}
    body = fonts.get("body", {}) if isinstance(fonts.get("body", {}), dict) else {}
    return {
        "headline": str(headline.get("category", "")).strip().lower(),
        "body": str(body.get("category", "")).strip().lower(),
    }


def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    hex_code = str(hex_code).strip().lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return h, s, v


def _contains_any(text: str, keywords: List[str]) -> bool:
    text_l = text.lower()
    return any(k in text_l for k in keywords)


def _is_reddish(hex_code: str) -> bool:
    h, _, _ = _rgb_to_hsv(_hex_to_rgb(hex_code))
    return h < 20 or h > 340


def _is_neon_like(hex_code: str) -> bool:
    _, s, v = _rgb_to_hsv(_hex_to_rgb(hex_code))
    return s >= 0.72 and v >= 0.82


def _rule_based_constraint_check(
    constraint: str,
    kit: Dict[str, Any],
    wcag_report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    text = constraint.strip()
    if not text:
        return None
    text_l = text.lower()
    palette = _extract_palette(kit)
    font_categories = _extract_font_categories(kit)

    if _contains_any(text_l, ["wcag", "accessibility", "accessible", "contrast", "colorblind"]):
        passed = bool(wcag_report.get("all_pass", False))
        return {
            "constraint": text,
            "status": "pass" if passed else "fail",
            "evidence": f"WCAG pass rate: {wcag_report.get('pass_rate', 0.0)}",
            "suggested_fix": (
                "" if passed else "Adjust palette until every color passes AA with readable text."
            ),
            "method": "rule_based",
        }

    if "no red" in text_l and palette:
        bad = [h for h in palette if _is_reddish(h)]
        passed = len(bad) == 0
        return {
            "constraint": text,
            "status": "pass" if passed else "fail",
            "evidence": "No red tones detected." if passed else f"Reddish colors found: {bad}",
            "suggested_fix": "" if passed else "Replace red hues with neutral or cooler alternatives.",
            "method": "rule_based",
        }

    if _contains_any(text_l, ["no neon", "avoid neon", "avoid harsh colors"]) and palette:
        neon = [h for h in palette if _is_neon_like(h)]
        passed = len(neon) == 0
        return {
            "constraint": text,
            "status": "pass" if passed else "fail",
            "evidence": "No neon-like colors detected." if passed else f"Neon-like colors found: {neon}",
            "suggested_fix": "" if passed else "Lower saturation and brightness for the highlighted colors.",
            "method": "rule_based",
        }

    if _contains_any(text_l, ["no serif", "avoid serif"]):
        has_serif = "serif" in {font_categories["headline"], font_categories["body"]}
        passed = not has_serif
        return {
            "constraint": text,
            "status": "pass" if passed else "fail",
            "evidence": (
                "No serif font category detected."
                if passed
                else f"Font categories include serif: {font_categories}"
            ),
            "suggested_fix": "" if passed else "Switch serif selections to sans-serif alternatives.",
            "method": "rule_based",
        }

    if _contains_any(text_l, ["no sans", "avoid sans"]):
        has_sans = "sans-serif" in {font_categories["headline"], font_categories["body"]}
        passed = not has_sans
        return {
            "constraint": text,
            "status": "pass" if passed else "fail",
            "evidence": (
                "No sans-serif font category detected."
                if passed
                else f"Font categories include sans-serif: {font_categories}"
            ),
            "suggested_fix": "" if passed else "Switch sans-serif selections to serif/display alternatives.",
            "method": "rule_based",
        }

    return None


def score_archetype_coherence(archetype: str, kit: Dict[str, Any]) -> Dict[str, Any]:
    client = _get_llm_client()
    if client is None:
        return {
            "score": 3.0,
            "summary": "No LLM key found. Used fallback neutral coherence score.",
            "strengths": [],
            "issues": ["Missing API key for LLM-based coherence scoring."],
            "method": "fallback",
        }

    prompt = f"""
You are a strict brand design reviewer.
Score how coherent this draft brand kit is with the chosen archetype.

Archetype: {archetype}
Draft kit JSON:
{json.dumps(kit, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "score": <float from 1.0 to 5.0>,
  "summary": "<1-2 sentence evaluation>",
  "strengths": ["item1", "item2"],
  "issues": ["issue1", "issue2"]
}}

Scoring rubric:
- 5.0: Fonts, palette, and tone strongly reinforce archetype.
- 4.0: Mostly aligned with small mismatches.
- 3.0: Partially aligned, clear inconsistencies.
- 2.0: Weak archetype fit.
- 1.0: Major mismatch.
"""

    try:
        resp = client.chat.completions.create(
            model=_coherence_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = _safe_json_loads(resp.choices[0].message.content, {})
    except Exception as exc:
        return {
            "score": 3.0,
            "summary": f"LLM coherence scorer failed: {exc}",
            "strengths": [],
            "issues": ["Coherence was not fully evaluated due to an API/runtime error."],
            "method": "fallback_error",
        }

    score_raw = parsed.get("score", 3.0)
    try:
        score = float(score_raw)
    except Exception:
        score = 3.0
    score = max(1.0, min(5.0, score))

    strengths = parsed.get("strengths", [])
    issues = parsed.get("issues", [])
    if not isinstance(strengths, list):
        strengths = []
    if not isinstance(issues, list):
        issues = []

    return {
        "score": round(score, 3),
        "summary": str(parsed.get("summary", "No summary provided.")).strip(),
        "strengths": [str(x) for x in strengths][:5],
        "issues": [str(x) for x in issues][:5],
        "method": "llm_judge",
    }


def _llm_constraint_review(
    constraints: List[str],
    state: Dict[str, Any],
    kit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    client = _get_llm_client()
    if client is None:
        return [
            {
                "constraint": c,
                "status": "uncertain",
                "evidence": "No API key found for LLM-based constraint review.",
                "suggested_fix": "Manually review this constraint against the generated kit.",
                "method": "fallback",
            }
            for c in constraints
        ]

    prompt = f"""
You are reviewing whether a generated brand kit satisfies user constraints.

Brand brief:
{state.get("brand_brief", "")}

Chosen archetype:
{state.get("archetype", "")}

Draft kit JSON:
{json.dumps(kit, ensure_ascii=False)}

Constraints to review:
{json.dumps(constraints, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "results": [
    {{
      "constraint": "<original constraint text>",
      "status": "pass|fail|uncertain",
      "evidence": "<brief evidence from kit>",
      "suggested_fix": "<actionable revision instruction>"
    }}
  ]
}}
"""

    try:
        resp = client.chat.completions.create(
            model=_constraint_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = _safe_json_loads(resp.choices[0].message.content, {})
    except Exception as exc:
        return [
            {
                "constraint": c,
                "status": "uncertain",
                "evidence": f"LLM constraint review failed: {exc}",
                "suggested_fix": "Retry with a stable API connection or review manually.",
                "method": "fallback_error",
            }
            for c in constraints
        ]

    raw_results = parsed.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    normalized: List[Dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        constraint = str(item.get("constraint", "")).strip()
        if not constraint:
            continue
        status = str(item.get("status", "uncertain")).strip().lower()
        if status not in {"pass", "fail", "uncertain"}:
            status = "uncertain"
        normalized.append(
            {
                "constraint": constraint,
                "status": status,
                "evidence": str(item.get("evidence", "")).strip(),
                "suggested_fix": str(item.get("suggested_fix", "")).strip(),
                "method": "llm_judge",
            }
        )

    by_constraint = {item["constraint"]: item for item in normalized}
    merged: List[Dict[str, Any]] = []
    for original in constraints:
        merged.append(
            by_constraint.get(
                original,
                {
                    "constraint": original,
                    "status": "uncertain",
                    "evidence": "Constraint not returned by LLM response.",
                    "suggested_fix": "Review manually and provide explicit revision instruction.",
                    "method": "llm_judge_missing",
                },
            )
        )
    return merged


def evaluate_constraints(
    state: Dict[str, Any],
    kit: Dict[str, Any],
    wcag_report: Dict[str, Any],
) -> Dict[str, Any]:
    constraints = _extract_constraints(state, kit)
    if not constraints:
        return {
            "all_pass": True,
            "pass_rate": 1.0,
            "pass_count": 0,
            "total": 0,
            "items": [],
        }

    reviewed: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for c in constraints:
        rule_result = _rule_based_constraint_check(c, kit, wcag_report)
        if rule_result is not None:
            reviewed.append(rule_result)
        else:
            unresolved.append(c)

    if unresolved:
        reviewed.extend(_llm_constraint_review(unresolved, state, kit))

    pass_count = sum(1 for item in reviewed if item.get("status") == "pass")
    total = len(reviewed)
    pass_rate = (pass_count / total) if total else 1.0
    all_pass = total == pass_count

    return {
        "all_pass": all_pass,
        "pass_rate": round(pass_rate, 3),
        "pass_count": pass_count,
        "total": total,
        "items": reviewed,
    }


def _compute_overall_score(
    wcag_report: Dict[str, Any],
    coherence_report: Dict[str, Any],
    constraint_report: Dict[str, Any],
) -> float:
    wcag_component = float(wcag_report.get("pass_rate", 0.0)) * 5.0
    coherence_component = float(coherence_report.get("score", 3.0))
    constraint_component = float(constraint_report.get("pass_rate", 0.0)) * 5.0
    weighted = (wcag_component * 0.4) + (coherence_component * 0.35) + (constraint_component * 0.25)
    return round(weighted, 3)


def _build_revision_feedback(
    wcag_report: Dict[str, Any],
    coherence_report: Dict[str, Any],
    constraint_report: Dict[str, Any],
) -> str:
    items: List[str] = []

    if not wcag_report.get("all_pass", False):
        failed_colors = [
            item.get("background")
            for item in wcag_report.get("color_checks", [])
            if not item.get("passes", False)
        ]
        items.append(
            "Accessibility: adjust palette colors to pass WCAG AA for text overlays. "
            f"Failing backgrounds: {failed_colors}."
        )

    if float(coherence_report.get("score", 0.0)) < COHERENCE_PASS_THRESHOLD:
        issues = coherence_report.get("issues", [])
        issue_text = "; ".join(str(x) for x in issues[:3]) if isinstance(issues, list) else ""
        items.append(
            "Archetype coherence: align font/style/palette more closely with the selected archetype. "
            f"Key issues: {issue_text}"
        )

    failed_constraints = [
        item
        for item in constraint_report.get("items", [])
        if item.get("status") in {"fail", "uncertain"}
    ]
    if failed_constraints:
        compact = []
        for item in failed_constraints[:6]:
            compact.append(
                f"[{item.get('constraint')}] -> {item.get('suggested_fix', 'Provide a direct fix.')}"
            )
        items.append("Constraint fixes: " + " | ".join(compact))

    if not items:
        items.append("No revision is required. The draft passes all QC checks.")

    lines = [f"{idx}) {text}" for idx, text in enumerate(items, start=1)]
    return "\n".join(lines)


def _select_best_kit_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_entry: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for item in history:
        if not isinstance(item, dict):
            continue
        scores = item.get("qc_scores", {})
        if not isinstance(scores, dict):
            continue
        score = float(scores.get("overall_score", 0.0))
        if score > best_score:
            best_score = score
            best_entry = item
    if best_entry and isinstance(best_entry.get("draft_brand_kit"), dict):
        return best_entry["draft_brand_kit"]
    return {}


def qc_agent(state: BrandMindState) -> BrandMindState:
    print("\n[QC] Starting quality review...")

    draft_brand_kit = state.get("draft_brand_kit")
    if not isinstance(draft_brand_kit, dict) or not draft_brand_kit:
        return {
            **state,
            "status": "failed",
            "qc_feedback": "Draft brand kit is missing. Generator must run before QC.",
            "qc_scores": {
                "overall_score": 0.0,
                "approved": False,
            },
        }

    palette_hex = _extract_palette(draft_brand_kit)
    wcag_report = evaluate_palette_wcag(palette_hex, level="AA", large_text=False)
    print(
        f"[QC] WCAG pass rate: {wcag_report.get('pass_rate')} "
        f"({wcag_report.get('pass_count')}/{wcag_report.get('total_checks')})"
    )

    coherence_report = score_archetype_coherence(state.get("archetype", ""), draft_brand_kit)
    print(f"[QC] Archetype coherence score: {coherence_report.get('score')}")

    constraint_report = evaluate_constraints(state, draft_brand_kit, wcag_report)
    print(
        f"[QC] Constraint pass rate: {constraint_report.get('pass_rate')} "
        f"({constraint_report.get('pass_count')}/{constraint_report.get('total')})"
    )

    overall_score = _compute_overall_score(wcag_report, coherence_report, constraint_report)
    approved = (
        bool(wcag_report.get("all_pass", False))
        and float(coherence_report.get("score", 0.0)) >= COHERENCE_PASS_THRESHOLD
        and bool(constraint_report.get("all_pass", False))
    )

    qc_scores = {
        "wcag": wcag_report,
        "coherence": coherence_report,
        "constraints": constraint_report,
        "overall_score": overall_score,
        "approved": approved,
    }

    feedback = None if approved else _build_revision_feedback(wcag_report, coherence_report, constraint_report)

    revision_history = list(state.get("revision_history") or [])
    current_iteration = int(state.get("iteration_count", 0))
    revision_history.append(
        {
            "iteration": current_iteration,
            "status": "approved" if approved else "failed",
            "qc_scores": qc_scores,
            "qc_feedback": feedback,
            "draft_brand_kit": draft_brand_kit,
        }
    )

    if approved:
        print("[QC] Draft approved.")
        return {
            **state,
            "status": "approved",
            "qc_feedback": None,
            "qc_scores": qc_scores,
            "approved_brand_kit": draft_brand_kit,
            "revision_history": revision_history,
        }

    next_iteration = current_iteration + 1
    if next_iteration >= MAX_ITERATIONS:
        best_kit = _select_best_kit_from_history(revision_history)
        max_iter_feedback = (
            f"{feedback}\n"
            "Max iterations reached. Returning the best-scoring draft collected so far."
        )
        print("[QC] Max iterations reached. Returning best available draft.")
        return {
            **state,
            "status": "failed",
            "iteration_count": next_iteration,
            "qc_feedback": max_iter_feedback,
            "qc_scores": qc_scores,
            "approved_brand_kit": best_kit if best_kit else draft_brand_kit,
            "revision_history": revision_history,
        }

    print("[QC] Draft failed. Sending revision feedback to Generator.")
    return {
        **state,
        "status": "generating",
        "iteration_count": next_iteration,
        "qc_feedback": feedback,
        "qc_scores": qc_scores,
        "revision_history": revision_history,
    }


if __name__ == "__main__":
    test_state: BrandMindState = {
        "brand_brief": (
            "Premium eco skincare for urban professionals. "
            "Needs WCAG accessible palette, no neon colors, and trustworthy modern tone."
        ),
        "clip_features": None,
        "archetype": "Organic",
        "archetype_rationale": "Nature-first premium identity.",
        "design_constraints": [
            "WCAG AA accessible palette",
            "No neon colors",
            "Trustworthy and modern visual tone",
        ],
        "design_spec": {
            "industry": "Skincare",
            "style_keywords": ["organic", "premium", "clean"],
        },
        "generator_trace": None,
        "draft_brand_kit": {
            "archetype": "Organic",
            "industry": "Skincare",
            "font_recommendation": {
                "headline": {"family": "Lora", "category": "serif"},
                "body": {"family": "Inter", "category": "sans-serif"},
            },
            "color_palette": {
                "hex_codes": ["#1F2937", "#F9FAFB", "#4B7F52", "#D97706", "#10B981"],
            },
            "design_rules": [
                "Use restrained contrast and natural tones.",
                "Maintain readability across touchpoints.",
            ],
        },
        "qc_feedback": None,
        "qc_scores": None,
        "iteration_count": 0,
        "status": "reviewing",
        "revision_history": [],
        "approved_brand_kit": None,
    }

    output = qc_agent(test_state)
    print("\n[QC] Output status:", output.get("status"))
    print("[QC] Approved:", output.get("qc_scores", {}).get("approved"))
    print("[QC] Overall score:", output.get("qc_scores", {}).get("overall_score"))
    if output.get("qc_feedback"):
        print("[QC] Feedback:\n", output["qc_feedback"])
