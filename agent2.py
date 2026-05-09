"""
Agent 2: Design Generator Agent

- Reads archetype + constraints from shared BrandMindState
- Infers a retrieval-friendly design spec
- Retrieves real font candidates from Google Fonts
- Retrieves design heuristic rules
- Converts heuristic rules into actionable generation constraints
- Retrieves/scored color palettes using user + heuristic constraints
- Assembles a structured draft brand kit
"""

from __future__ import annotations

import colorsys
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from state import BrandMindState
from tools.font_lookup import font_lookup
from tools.color_retrieve import color_retrieve
from tools.heuristic_search import (
    heuristic_search,
    heuristics_to_generation_constraints,
)


# ─────────────────────────────────────────────────────────────────────────────
# Color utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_hsv(hex_code: str):
    s = str(hex_code).strip().lstrip("#")
    r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(
        h % 1.0,
        max(0.0, min(1.0, s)),
        max(0.0, min(1.0, v)),
    )
    return "#" + "".join(f"{int(round(c * 255)):02X}" for c in (r, g, b))


def _is_neon(hex_code: str) -> bool:
    _, s, v = _hex_to_hsv(hex_code)
    return s >= 0.72 and v >= 0.82


def _is_reddish(hex_code: str) -> bool:
    h, _, _ = _hex_to_hsv(hex_code)
    deg = h * 360.0
    return deg < 20.0 or deg > 340.0


def _is_warm_or_earthy(hex_code: str) -> bool:
    h, s, v = _hex_to_hsv(hex_code)
    deg = h * 360.0
    warm_hue = 0 <= deg <= 65
    earthy = 10 <= deg <= 55 and 0.20 <= s <= 0.75 and v <= 0.75
    return warm_hue or earthy


def _desaturate_neon(hex_code: str) -> str:
    h, s, v = _hex_to_hsv(hex_code)
    if s >= 0.72:
        s = 0.65
    if v >= 0.82:
        v = 0.78
    return _hsv_to_hex(h, s, v)


def _shift_red_to_amber(hex_code: str) -> str:
    h, s, v = _hex_to_hsv(hex_code)
    deg = h * 360.0
    if deg < 20.0 or deg > 340.0:
        h = 30.0 / 360.0
    return _hsv_to_hex(h, s, v)


def _shift_warm_to_cool(hex_code: str) -> str:
    _, s, v = _hex_to_hsv(hex_code)
    h = 210.0 / 360.0
    s = min(s, 0.55)
    v = max(0.25, min(v, 0.85))
    return _hsv_to_hex(h, s, v)


def repair_palette(hex_codes: List[str], constraints: List[str]) -> List[str]:
    if not hex_codes:
        return hex_codes

    constraint_text = " ".join(str(c).lower() for c in constraints)

    avoid_red = "no red" in constraint_text or "avoid red" in constraint_text
    avoid_neon = (
        "no neon" in constraint_text
        or "avoid neon" in constraint_text
        or "avoid high-saturation" in constraint_text
        or "avoid high saturation" in constraint_text
        or "neon-like" in constraint_text
    )
    avoid_warm_earthy = (
        "no warm" in constraint_text
        or "avoid warm" in constraint_text
        or "no earthy" in constraint_text
        or "avoid earthy" in constraint_text
        or "no warm or earthy" in constraint_text
    )

    out: List[str] = []

    for code in hex_codes:
        c = str(code).strip()
        if not c:
            continue
        if not c.startswith("#"):
            c = "#" + c
        if len(c) != 7:
            continue
        try:
            _hex_to_hsv(c)
        except Exception:
            continue

        if avoid_red and _is_reddish(c):
            c = _shift_red_to_amber(c)
        if avoid_warm_earthy and _is_warm_or_earthy(c):
            c = _shift_warm_to_cool(c)
        if avoid_neon and _is_neon(c):
            c = _desaturate_neon(c)

        out.append(c.upper())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# LLM provider helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in ("openai", "groq"):
        return explicit
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    return "openai"


def _get_client() -> OpenAI:
    if _resolve_provider() == "groq":
        return OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
    )


def _get_model() -> str:
    if _resolve_provider() == "groq":
        return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


client = _get_client()


# ─────────────────────────────────────────────────────────────────────────────
# Archetype tone injection
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_TONE_LEXICON = {
    "corporate": ["trustworthy", "reliable", "professional", "established", "authoritative"],
    "tech": ["precise", "modern", "innovative", "intelligent", "efficient"],
    "minimal": ["simple", "intentional", "refined", "uncluttered", "deliberate"],
    "organic": ["natural", "calming", "grounded", "sustainable", "gentle"],
    "luxury": ["premium", "refined", "timeless", "exclusive", "sophisticated"],
    "playful": ["energetic", "joyful", "lighthearted", "spontaneous", "fun"],
    "bold": ["confident", "decisive", "daring", "assertive", "dynamic"],
    "artisan": ["handmade", "authentic", "warm", "craft-driven", "neighborly"],
    "heritage": ["traditional", "enduring", "rooted", "established", "time-honored"],
    "youthful": ["vibrant", "fresh", "optimistic", "lively", "energetic"],
}


# ─────────────────────────────────────────────────────────────────────────────
# MODIFIED: Multi-group fallback palettes (3 variants per archetype)
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_FALLBACK_PALETTES = {
    "corporate": [
        ["#0A1628", "#FFFFFF", "#2E5090", "#F4F6F9", "#111111"],
        ["#1B2A4A", "#F0F4FF", "#3A5FA0", "#E8ECF4", "#0D1B2A"],
        ["#0F2044", "#FFFFFF", "#4A6FA5", "#EEF2F7", "#1C2D4A"],
    ],
    "tech": [
        ["#0D1117", "#161B22", "#21262D", "#58A6FF", "#FFFFFF"],
        ["#0A0E1A", "#1E2A3A", "#2E4A6A", "#4FC3F7", "#FFFFFF"],
        ["#050D18", "#0D2137", "#1A3A5C", "#38BDF8", "#F0F9FF"],
    ],
    "minimal": [
        ["#1A1A1A", "#FFFFFF", "#F5F5F5", "#333333", "#888888"],
        ["#111111", "#FAFAFA", "#E8E8E8", "#444444", "#999999"],
        ["#0D0D0D", "#F8F8F8", "#EFEFEF", "#555555", "#AAAAAA"],
    ],
    "organic": [
        ["#3B5249", "#519872", "#A4C3A2", "#F0EAD6", "#8B5E3C"],
        ["#2D4A3E", "#4A7C59", "#8FAF8C", "#EDE8D0", "#7A4F2E"],
        ["#1F3528", "#3D6647", "#7A9E78", "#F5F0E0", "#6B3F20"],
    ],
    "luxury": [
        ["#1C1C1C", "#B8960C", "#FFFFFF", "#2C2C2C", "#D4AF37"],
        ["#0D0D0D", "#C9A227", "#F5F5F5", "#1A1A1A", "#E8C84A"],
        ["#141414", "#A07C10", "#FAFAFA", "#242424", "#BF9B30"],
    ],
    "playful": [
        ["#FF6B6B", "#FFE66D", "#4ECDC4", "#FFFFFF", "#2C3E50"],
        ["#FF8E8E", "#FFD93D", "#6BCFCA", "#FFFFFF", "#354A5E"],
        ["#FF5252", "#FFCA28", "#26C6DA", "#FFFFFF", "#263238"],
    ],
    "bold": [
        ["#E63946", "#1D3557", "#FFFFFF", "#457B9D", "#F1FAEE"],
        ["#D62839", "#14213D", "#FFFFFF", "#3A6B8A", "#E8F4EA"],
        ["#C1121F", "#0D1B2A", "#FFFFFF", "#2E5D7E", "#F0F7F0"],
    ],
    "artisan": [
        ["#6B4226", "#D4A373", "#FEFAE0", "#CCD5AE", "#E9EDC9"],
        ["#5C3820", "#C89060", "#FDF8E0", "#BDC9A0", "#DCEAB8"],
        ["#4A2C18", "#BC7D4D", "#FEF6E0", "#AEC090", "#D0E4A8"],
    ],
    "heritage": [
        ["#2C1810", "#8B4513", "#D2691E", "#F5DEB3", "#FFFFFF"],
        ["#1E0F08", "#7A3A10", "#C05A18", "#EDD09E", "#F8F0E0"],
        ["#140A04", "#6A2D0D", "#B04E14", "#E5C890", "#F5EBD8"],
    ],
    "youthful": [
        ["#FF6B9D", "#C44569", "#F8B500", "#00B4D8", "#FFFFFF"],
        ["#FF85AD", "#D45579", "#FFCA00", "#00C8E8", "#FFFFFF"],
        ["#FF4D8D", "#B83560", "#F0A800", "#00A0C8", "#FAFAFA"],
    ],
}

WCAG_ANCHORS = ["#000000", "#FFFFFF"]


def _inject_archetype_tokens(
    archetype: str,
    existing: List[str],
    cap: int = 8,
) -> List[str]:
    enabled = os.environ.get("BRANDMIND_TONE_INJECTION", "1").lower() not in (
        "0", "false", "no",
    )
    if not enabled:
        return list(existing or [])[:cap]

    canonical = ARCHETYPE_TONE_LEXICON.get((archetype or "").strip().lower(), [])
    if not canonical:
        return list(existing or [])[:cap]

    seen: set[str] = set()
    merged: List[str] = []
    for token in list(canonical) + list(existing or []):
        norm = str(token).strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            merged.append(str(token).strip())

    return merged[:cap]


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: infer design spec
# MODIFIED: accepts state to extract prev_hex_list for feedback injection
# ─────────────────────────────────────────────────────────────────────────────

def infer_design_spec(
    brand_brief: str,
    archetype: str,
    constraints: List[str],
    clip_context: str = "",
    qc_feedback: str = "",
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else "- None"
    clip_section = f"\nVisual context: {clip_context}\n" if clip_context else ""

    # Collect previously failed hex codes from revision history
    prev_hex_list: List[str] = []
    if state:
        for entry in state.get("revision_history", []):
            prev_kit = entry.get("draft_brand_kit", {})
            prev_hex = (prev_kit.get("color_palette") or {}).get("hex_codes", [])
            prev_hex_list.extend(prev_hex)
        prev_hex_list = list(set(prev_hex_list))

    # MODIFIED: more forceful feedback section with explicit failed hex codes
    feedback_section = (
        f"""
REVISION MODE - The previous draft FAILED QC. You MUST produce a different output.

QC failure details:
{qc_feedback}

Colors that FAILED and must NOT be reused: {prev_hex_list}
You MUST change your primary_emotions, style_keywords, and palette_notes
to produce a visually different result. Do not repeat the same values as before.
"""
        if qc_feedback
        else ""
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

    try:
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        spec = _safe_json_loads(resp.choices[0].message.content)
        print(f"[Generator] infer_design_spec OK: industry={spec.get('industry')}")
    except Exception as exc:
        print(f"[Generator] infer_design_spec FAILED: {exc}")
        spec = {}

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
            f"The palette should reinforce a {archetype.lower()} brand feeling.",
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: font pairing
# ─────────────────────────────────────────────────────────────────────────────

def choose_font_pair(font_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not font_candidates:
        return {
            "headline_font": None,
            "body_font": None,
            "rationale": "No font candidates returned.",
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
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: heuristic retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_design_heuristics(
    archetype: str,
    design_spec: Dict[str, Any],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    heuristics: List[Dict[str, Any]] = []
    weights = state.get("heuristic_weights")

    heuristics.extend(heuristic_search(archetype, weights=weights, top_k=3))

    for attr in design_spec.get("brand_attributes", []):
        heuristics.extend(heuristic_search(attr, weights=weights, top_k=2))

    seen_rule_ids: set = set()
    deduped: List[Dict[str, Any]] = []
    for rule in heuristics:
        rid = rule.get("id")
        if rid and rid not in seen_rule_ids:
            seen_rule_ids.add(rid)
            deduped.append(rule)

    seen_rule_texts: set = set()
    final: List[Dict[str, Any]] = []
    for item in deduped:
        rule_text = item.get("rule", "").strip()
        if rule_text and rule_text not in seen_rule_texts:
            seen_rule_texts.add(rule_text)
            final.append(item)

    return final[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: assemble final draft brand kit
# ─────────────────────────────────────────────────────────────────────────────

def assemble_draft_brand_kit(
    archetype: str,
    design_spec: Dict[str, Any],
    font_pair: Dict[str, Any],
    palette_result: Dict[str, Any],
    heuristics: List[Dict[str, Any]],
    constraints: List[str],
    heuristic_constraints: Dict[str, List[str]],
) -> Dict[str, Any]:
    palette = palette_result.get("best_palette", {})
    rules = [h.get("rule") for h in heuristics if h.get("rule")]

    enriched_tone_keywords = _inject_archetype_tokens(
        archetype, design_spec.get("tone_keywords", []),
    )
    enriched_brand_attrs = _inject_archetype_tokens(
        archetype, design_spec.get("brand_attributes", []),
    )

    return {
        "archetype": archetype,
        "industry": design_spec["industry"],
        "target_audience": design_spec["target_audience"],
        "primary_emotions": design_spec["primary_emotions"],
        "style_keywords": design_spec["style_keywords"],
        "brand_attributes": enriched_brand_attrs,
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
            "palette_rationale": (
                palette.get("palette_rationale") or palette_result.get("rationale")
            ),
        },
        "design_rules": rules,
        "constraints_checked_later": constraints,
        "tone_and_voice_seed": {
            "tone_keywords": enriched_tone_keywords,
            "palette_notes": design_spec["palette_notes"],
        },
        "generator_trace": {
            "font_candidates_seen": font_pair.get("headline_font"),
            "palette_candidates_seen": palette_result.get("top_k_palettes", []),
            "heuristics_used": heuristics,
            "heuristic_generation_constraints": heuristic_constraints,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional narrative rationale
# ─────────────────────────────────────────────────────────────────────────────

def generate_archetype_alignment(archetype: str, kit: Dict[str, Any]) -> str:
    headline = (kit.get("font_recommendation") or {}).get("headline") or {}
    body = (kit.get("font_recommendation") or {}).get("body") or {}
    palette_hex = (kit.get("color_palette") or {}).get("hex_codes") or []
    tone_keywords = (kit.get("tone_and_voice_seed") or {}).get("tone_keywords") or []

    prompt = (
        "You are a senior brand designer writing a one-paragraph design rationale "
        "for a brand book. In 2-3 sentences (max 80 words), explain how the chosen "
        f"font pairing, color palette, and tone collectively express the {archetype} "
        "brand archetype. Be specific about which choice does what. No marketing "
        "fluff, no lists, no bullet points.\n\n"
        f"Archetype: {archetype}\n"
        f"Headline font: {headline.get('family')} ({headline.get('category')})\n"
        f"Body font: {body.get('family')} ({body.get('category')})\n"
        f"Palette: {palette_hex}\n"
        f"Tone keywords: {tone_keywords}\n\n"
        'Return ONLY valid JSON: {"rationale": "<2-3 sentences>"}'
    )

    try:
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        parsed = _safe_json_loads(resp.choices[0].message.content)
        text = str(parsed.get("rationale", "")).strip()
        if text:
            return text
    except Exception as exc:
        print(f"[Generator] archetype_alignment narrative FAILED: {exc}")

    return (
        f"This {archetype} kit pairs the chosen typography and palette to reinforce "
        f"the archetype's core register."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent 2 node
# ─────────────────────────────────────────────────────────────────────────────

def design_generator_agent(state: BrandMindState) -> BrandMindState:
    print("\n[Generator] Starting design generation...")

    brand_brief = state["brand_brief"]
    archetype = state["archetype"]
    constraints = state.get("design_constraints") or state.get("constraints") or []
    clip_context = state.get("clip_context", "") or state.get("clip_features", "")
    qc_feedback = state.get("qc_feedback", "")

    # 1. Infer retrieval-friendly design spec
    # MODIFIED: pass state so infer_design_spec can read revision_history
    design_spec = infer_design_spec(
        brand_brief=brand_brief,
        archetype=archetype,
        constraints=constraints,
        clip_context=clip_context,
        qc_feedback=qc_feedback,
        state=state,
    )

    print(f"[Generator] Design spec industry: {design_spec.get('industry')}")
    print(f"[Generator] Design spec emotions: {design_spec.get('primary_emotions')}")
    print(f"[Generator] Design spec attributes: {design_spec.get('brand_attributes')}")

    # 2. Retrieve fonts
    font_candidates = font_lookup(
        archetype=archetype,
        style=design_spec["font_style"],
        top_k=8,
    )
    font_pair = choose_font_pair(font_candidates)

    print(
        "[Generator] Font pair: "
        f"{(font_pair.get('headline_font') or {}).get('family')} + "
        f"{(font_pair.get('body_font') or {}).get('family')}"
    )

    # 3. Exclude palette colors from previous failed drafts
    excluded_hex: List[str] = []
    revision_history = state.get("revision_history", [])

    for entry in revision_history:
        prev_kit = entry.get("draft_brand_kit", {})
        prev_hex = (prev_kit.get("color_palette") or {}).get("hex_codes", [])
        for h in prev_hex:
            if h not in excluded_hex:
                excluded_hex.append(h)

    print(f"[Generator] Excluding {len(excluded_hex)} hex codes from previous drafts.")

    # 4. Retrieve heuristics BEFORE palette retrieval
    heuristics = retrieve_design_heuristics(
        archetype=archetype,
        design_spec=design_spec,
        state=state,
    )

    heuristic_constraints = heuristics_to_generation_constraints(heuristics)
    combined_constraints = constraints + heuristic_constraints.get("palette_constraints", [])

    print(f"[Generator] Retrieved {len(heuristics)} heuristic rule(s).")
    print(
        "[Generator] Heuristic palette constraints: "
        f"{heuristic_constraints.get('palette_constraints', [])}"
    )

    # 5. Retrieve palette using both user constraints and heuristic-derived constraints
    palette_result = color_retrieve(
        emotions=design_spec["primary_emotions"],
        industry=design_spec["industry"],
        style_keywords=design_spec["style_keywords"],
        constraints=combined_constraints,
        top_k=5,
        excluded_hex=excluded_hex,
    )

    best = palette_result.get("best_palette", {})
    print(
        "[Generator] Best palette: "
        f"score={best.get('total_score', 0):.2f}, "
        f"emotion_score={best.get('emotion_score', 0):.2f}"
    )

    # 6. MODIFIED: fallback rotates by iteration_count, filters excluded, appends WCAG anchors
    if best.get("total_score", 0) < 3.0 or (
        archetype.lower() in ARCHETYPE_FALLBACK_PALETTES
        and best.get("emotion_score", 0) == 0.0
    ):
        iteration = int(state.get("iteration_count", 0))
        candidates = ARCHETYPE_FALLBACK_PALETTES.get(
            archetype.lower(),
            [["#1A1A1A", "#FFFFFF", "#2E5090", "#F4F6F9", "#4A90D9"]],
        )

        # candidates is now a list of lists
        if isinstance(candidates[0], list):
            fallback_hex = list(candidates[iteration % len(candidates)])
        else:
            fallback_hex = list(candidates)

        # Remove colors that were already used and failed
        fallback_hex = [h for h in fallback_hex if h not in excluded_hex]
        if not fallback_hex:
            fallback_hex = list(candidates[(iteration + 1) % len(candidates)])

        # Append WCAG-safe anchors to guarantee at least one high-contrast pair
        for anchor in WCAG_ANCHORS:
            if anchor not in fallback_hex:
                fallback_hex.append(anchor)

        print(
            "[Generator] Low-quality palette detected, "
            f"using archetype fallback (iteration {iteration}): {fallback_hex}"
        )

        palette_result["best_palette"] = {
            **best,
            "hex_codes": fallback_hex,
            "palette_name": f"{archetype}_archetype_fallback_iter{iteration}",
            "palette_rationale": (
                f"Archetype-safe fallback palette for {archetype} "
                f"(iteration {iteration})."
            ),
        }

    # 7. Deterministic repair using user + heuristic constraints
    current_best = palette_result.get("best_palette", {})
    raw_hex = current_best.get("hex_codes", [])
    repaired_hex = repair_palette(raw_hex, combined_constraints)

    if repaired_hex != raw_hex:
        print(f"[Generator] Palette repair applied: {raw_hex} -> {repaired_hex}")
        palette_result["best_palette"] = {
            **current_best,
            "hex_codes": repaired_hex,
        }

    # 8. Assemble draft brand kit
    draft_brand_kit = assemble_draft_brand_kit(
        archetype=archetype,
        design_spec=design_spec,
        font_pair=font_pair,
        palette_result=palette_result,
        heuristics=heuristics,
        constraints=constraints,
        heuristic_constraints=heuristic_constraints,
    )

    # 9. Optional one-paragraph design rationale
    if os.environ.get("BRANDMIND_NARRATIVE", "1").lower() not in ("0", "false", "no"):
        draft_brand_kit["archetype_alignment"] = generate_archetype_alignment(
            archetype,
            draft_brand_kit,
        )

    # 10. Write back to shared state
    state["design_spec"] = design_spec
    state["draft_brand_kit"] = draft_brand_kit
    state["generator_output"] = {
        "status": "draft_ready",
        "message": (
            "Agent 2 assembled a draft brand kit using retrieved fonts, "
            "colors, and design rules."
        ),
    }

    print("[Generator] Draft brand kit ready.")

    return state


# Compatibility alias
generator_agent = design_generator_agent


# ─────────────────────────────────────────────────────────────────────────────
# Manual smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_state: BrandMindState = {
        "brand_brief": (
            "Nimbus Ledger is a B2B fintech platform for cross-border invoices. "
            "The brand identity should feel trustworthy, precise, and modern for "
            "enterprise finance teams. The color palette must be WCAG AA accessible. "
            "Avoid neon colors. Avoid serif fonts."
        ),
        "clip_features": None,
        "archetype": "Tech",
        "archetype_rationale": "A technical fintech product fits the Tech archetype.",
        "design_constraints": [
            "Color palette must be WCAG AA accessible",
            "No neon colors",
            "No serif fonts",
        ],
        "constraints": [
            "Color palette must be WCAG AA accessible",
            "No neon colors",
            "No serif fonts",
        ],
        "draft_brand_kit": None,
        "qc_feedback": None,
        "qc_scores": None,
        "heuristic_weights": None,
        "iteration_count": 0,
        "status": "generating",
        "revision_history": [],
        "approved_brand_kit": None,
    }

    output = design_generator_agent(test_state)
    print(json.dumps(output["draft_brand_kit"], indent=2, ensure_ascii=False))
