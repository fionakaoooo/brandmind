"""
Agent 2: Design Generator Agent
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from state import BrandMindState
from tools.font_lookup import font_lookup
from tools.color_retrieve import color_retrieve
from tools.heuristic_search import heuristic_search


client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


ARCHETYPE_FALLBACK_PALETTES = {
    "corporate": ["#0A1628", "#FFFFFF", "#2E5090", "#F4F6F9", "#111111"],
    "tech":      ["#0D1117", "#161B22", "#21262D", "#58A6FF", "#FFFFFF"],
    "minimal":   ["#1A1A1A", "#FFFFFF", "#F5F5F5", "#333333", "#888888"],
    "organic":   ["#3B5249", "#519872", "#A4C3A2", "#F0EAD6", "#8B5E3C"],
    "luxury":    ["#1C1C1C", "#B8960C", "#FFFFFF", "#2C2C2C", "#D4AF37"],
    "playful":   ["#FF6B6B", "#FFE66D", "#4ECDC4", "#FFFFFF", "#2C3E50"],
    "bold":      ["#E63946", "#1D3557", "#FFFFFF", "#457B9D", "#F1FAEE"],
    "artisan":   ["#6B4226", "#D4A373", "#FEFAE0", "#CCD5AE", "#E9EDC9"],
    "heritage":  ["#2C1810", "#8B4513", "#D2691E", "#F5DEB3", "#FFFFFF"],
    "youthful":  ["#FF6B9D", "#C44569", "#F8B500", "#00B4D8", "#FFFFFF"],
}


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def infer_design_spec(
    brand_brief: str,
    archetype: str,
    constraints: List[str],
    clip_context: str = "",
    qc_feedback: str = "",
) -> Dict[str, Any]:

    constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else "- None"
    clip_section = f"\nVisual context: {clip_context}\n" if clip_context else ""
    feedback_section = (
        f"""
REVISION MODE - You MUST address all of the following QC failures.
Each item below is a constraint that the previous draft FAILED.
Your output spec must directly fix every listed issue:
{qc_feedback}
"""
        if qc_feedback else ""
    )

    prompt = f"""
You are a senior brand design strategist.

Given the brand brief, the already-selected archetype, and extracted constraints,
produce a structured design spec for retrieval-based brand generation.

Brand archetype: {archetype}

Brand brief:
{brand_brief}

Constraints:
{constraint_text}
{clip_section}{feedback_section}
Return ONLY valid JSON in this schema:
{{
  "industry": "<one short phrase>",
  "target_audience": "<one short phrase>",
  "primary_emotions": ["emotion1", "emotion2", "emotion3"],
  "style_keywords": ["keyword1", "keyword2", "keyword3"],
  "font_style": "<short style phrase for font lookup>",
  "brand_attributes": ["attribute1", "attribute2", "attribute3"],
  "tone_keywords": ["tone1", "tone2", "tone3"],
  "palette_notes": "<1 sentence about what the palette should feel like>"
}}

Rules:
- Keep emotions concrete and retrieval-friendly, e.g. elegant, playful, calm, modern, energetic, organic.
- Use 3 primary emotions max.
- font_style should be usable for font retrieval, e.g. "modern sans serif", "high contrast serif", "rounded friendly sans".
- brand_attributes should be concise because they will be used for heuristic search.
- If QC feedback is provided above, adjust your spec to directly address those issues.
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    spec = _safe_json_loads(resp.choices[0].message.content)

    return {
        "industry": spec.get("industry", "general"),
        "target_audience": spec.get("target_audience", "general audience"),
        "primary_emotions": spec.get("primary_emotions", [archetype.lower()])[:3],
        "style_keywords": spec.get("style_keywords", [archetype.lower()])[:5],
        "font_style": spec.get("font_style", archetype.lower()),
        "brand_attributes": spec.get("brand_attributes", [archetype.lower()])[:5],
        "tone_keywords": spec.get("tone_keywords", [archetype.lower()])[:5],
        "palette_notes": spec.get(
            "palette_notes",
            f"The palette should reinforce a {archetype.lower()} brand feeling."
        ),
    }


def choose_font_pair(font_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not font_candidates:
        return {
            "headline_font": None,
            "body_font": None,
            "rationale": "No font candidates returned."
        }

    headline = font_candidates[0]
    body = None

    for cand in font_candidates[1:]:
        if cand.get("category") != headline.get("category"):
            body = cand
            break

    if body is None:
        body = font_candidates[1] if len(font_candidates) > 1 else font_candidates[0]

    return {
        "headline_font": headline,
        "body_font": body,
        "rationale": (
            f"Selected {headline.get('family')} as the expressive/display face and "
            f"{body.get('family')} as the supporting/body face."
        )
    }


def assemble_draft_brand_kit(
    archetype: str,
    design_spec: Dict[str, Any],
    font_pair: Dict[str, Any],
    palette_result: Dict[str, Any],
    heuristics: List[Dict[str, Any]],
    constraints: List[str],
) -> Dict[str, Any]:
    palette = palette_result.get("best_palette", {})
    rules = [h.get("rule") for h in heuristics if h.get("rule")]

    return {
        "archetype": archetype,
        "industry": design_spec["industry"],
        "target_audience": design_spec["target_audience"],
        "primary_emotions": design_spec["primary_emotions"],
        "style_keywords": design_spec["style_keywords"],
        "font_recommendation": {
            "headline": font_pair.get("headline_font"),
            "body": font_pair.get("body_font"),
            "pairing_rationale": font_pair.get("rationale"),
        },
        "color_palette": {
            "palette_name": palette.get("palette_name"),
            "hex_codes": palette.get("hex_codes", []),
            "matched_emotions": palette.get("matched_emotions", []),
            "industry_bonus": palette.get("industry_bonus", 0.0),
            "emotion_score": palette.get("emotion_score", 0.0),
            "emoset_alignment": palette.get("emoset_alignment", {}),
            "palette_rationale": palette_result.get("rationale"),
        },
        "design_rules": rules,
        "constraints_checked_later": constraints,
        "tone_and_voice_seed": {
            "tone_keywords": design_spec["tone_keywords"],
            "palette_notes": design_spec["palette_notes"],
        },
        "generator_trace": {
            "font_candidates_seen": font_pair.get("headline_font"),
            "palette_candidates_seen": palette_result.get("top_k_palettes", []),
            "heuristics_used": heuristics,
        }
    }


def design_generator_agent(state: BrandMindState) -> BrandMindState:
    brand_brief = state["brand_brief"]
    archetype = state["archetype"]
    constraints = state.get("design_constraints", [])
    clip_context = state.get("clip_context", "")
    qc_feedback = state.get("qc_feedback", "")

    design_spec = infer_design_spec(
        brand_brief=brand_brief,
        archetype=archetype,
        constraints=constraints,
        clip_context=clip_context,
        qc_feedback=qc_feedback,
    )

    font_candidates = font_lookup(
        archetype=archetype,
        style=design_spec["font_style"],
        top_k=8
    )
    font_pair = choose_font_pair(font_candidates)

    excluded_hex: List[str] = []
    if qc_feedback and state.get("draft_brand_kit"):
        prev_palette = state["draft_brand_kit"].get("color_palette", {})
        excluded_hex = prev_palette.get("hex_codes", [])
    print(f"[Generator] Excluding {len(excluded_hex)} hex codes from previous draft.")

    palette_result = color_retrieve(
        emotions=design_spec["primary_emotions"],
        industry=design_spec["industry"],
        style_keywords=design_spec["style_keywords"],
        constraints=constraints,
        top_k=5,
        excluded_hex=excluded_hex,
    )

    # 新增：调色板质量检查，低质量时 fallback 到原型安全调色板
    best = palette_result.get("best_palette", {})
    if best.get("total_score", 0) < 3.0 or (
        archetype.lower() in ARCHETYPE_FALLBACK_PALETTES
        and best.get("emotion_score", 0) == 0.0
    ):
        fallback_hex = ARCHETYPE_FALLBACK_PALETTES.get(
            archetype.lower(),
            ["#1A1A1A", "#FFFFFF", "#2E5090", "#F4F6F9", "#4A90D9"]
        )
        print(f"[Generator] Low-quality palette detected (score={best.get('total_score', 0):.2f}), using archetype fallback: {fallback_hex}")
        palette_result["best_palette"] = {
            **best,
            "hex_codes": fallback_hex,
            "palette_name": f"{archetype}_archetype_fallback",
            "palette_rationale": f"Archetype-safe fallback palette for {archetype}.",
        }

    heuristics: List[Dict[str, Any]] = []
    for attr in design_spec["brand_attributes"]:
        heuristics.extend(heuristic_search(attr))

    seen = set()
    deduped_heuristics = []
    for item in heuristics:
        rule = item.get("rule", "").strip()
        if rule and rule not in seen:
            seen.add(rule)
            deduped_heuristics.append(item)

    draft_brand_kit = assemble_draft_brand_kit(
        archetype=archetype,
        design_spec=design_spec,
        font_pair=font_pair,
        palette_result=palette_result,
        heuristics=deduped_heuristics[:6],
        constraints=constraints,
    )

    state["design_spec"] = design_spec
    state["draft_brand_kit"] = draft_brand_kit
    state["generator_output"] = {
        "status": "draft_ready",
        "message": "Agent 2 assembled a draft brand kit using retrieved fonts, colors, and design rules."
    }

    return state
