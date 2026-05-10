from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph import run_pipeline
from state import ARCHETYPES
from tools.color_retrieve import color_retrieve
from tools.font_lookup import font_lookup
from tools.heuristic_search import heuristic_search
from tools.wcag_check import evaluate_palette_wcag, evaluate_palette_wcag_min_pairs


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass
class BenchmarkCase:
    case_id: str
    brief: str
    constraints: List[str]


BENCHMARK_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        case_id="case_01_fintech_corporate",
        brief=(
            "Nimbus Ledger is a B2B fintech platform for cross-border invoices. "
            "Brand identity should feel trustworthy, precise, and modern for enterprise finance teams."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "Avoid serif fonts",
        ],
    ),
    BenchmarkCase(
        case_id="case_02_kids_playful",
        brief=(
            "Pebble Pop is a toy subscription startup for kids age 6-10. "
            "The brand should feel playful and energetic while staying readable for parents."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No red",
        ],
    ),
    BenchmarkCase(
        case_id="case_03_luxury_jewelry",
        brief=(
            "Etoile Vault is a luxury jewelry house focused on heirloom craftsmanship. "
            "The style should feel premium, refined, and timeless."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No sans",
        ],
    ),
    BenchmarkCase(
        case_id="case_04_organic_skincare",
        brief=(
            "Verdant Dew is an eco skincare brand for urban professionals seeking clean ingredients. "
            "The brand should feel organic, calm, and trustworthy."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No red",
        ],
    ),
    BenchmarkCase(
        case_id="case_05_cybersecurity_tech",
        brief=(
            "ShieldMesh is a cybersecurity SaaS platform for engineering teams. "
            "Identity should feel technical, secure, and modern."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "Avoid serif fonts",
        ],
    ),
    BenchmarkCase(
        case_id="case_06_artisan_bakery",
        brief=(
            "Oak Crumb is an artisan sourdough bakery emphasizing handmade quality and neighborhood warmth. "
            "The brand should feel authentic, human, and craft-driven."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No sans",
        ],
    ),
    BenchmarkCase(
        case_id="case_07_heritage_whiskey",
        brief=(
            "Glen Morrow is a Scottish whiskey brand targeting affluent men aged 35-60. "
            "The brand should feel classic, heritage-driven, and sophisticated. "
            "Think old-world craftsmanship and timeless tradition. No modern or tech aesthetics."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No sans",
            "No modern or tech aesthetics",
        ],
    ),
    BenchmarkCase(
        case_id="case_08_bold_streetwear",
        brief=(
            "CRUX is a streetwear brand targeting urban youth aged 16-25. "
            "The brand should feel bold, rebellious, and high-energy. "
            "Use high contrast colors. No corporate or bland aesthetics."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "High contrast colors required",
            "No corporate aesthetics",
            "Avoid serif fonts",
        ],
    ),
    BenchmarkCase(
        case_id="case_09_minimal_architecture",
        brief=(
            "Forma Studio is an architecture firm targeting premium residential clients. "
            "The brand should feel minimal, refined, and intellectually serious. "
            "Monochromatic palette preferred. No decorative or playful elements."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No decorative elements",
            "Monochromatic or near-monochromatic palette",
        ],
    ),
    BenchmarkCase(
        case_id="case_10_playful_pet_brand",
        brief=(
            "Pawsome is a pet food and accessories brand targeting millennial pet owners. "
            "The brand should feel fun, warm, and trustworthy. "
            "Bright friendly colors. No clinical or corporate feel."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No red",
            "No clinical or corporate feel",
        ],
    ),
    BenchmarkCase(
        case_id="case_11_corporate_consulting",
        brief=(
            "Meridian Advisory is a management consulting firm targeting Fortune 500 companies. "
            "The brand should feel authoritative, intelligent, and trustworthy. "
            "Conservative navy and gray palette. No playful or casual elements."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "Avoid serif fonts",
            "No playful elements",
        ],
    ),
    BenchmarkCase(
        case_id="case_12_youthful_edtech",
        brief=(
            "Spark is an online learning platform targeting college students aged 18-24. "
            "The brand should feel energetic, modern, and approachable. "
            "Vibrant colors welcome. Must feel inclusive and motivating."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "Avoid serif fonts",
            "Must feel inclusive and energetic",
        ],
    ),
    BenchmarkCase(
        case_id="case_13_organic_restaurant",
        brief=(
            "Terroir is a farm-to-table restaurant targeting food-conscious adults aged 28-50. "
            "The brand should feel earthy, warm, and artisanal. "
            "Muted natural tones. No synthetic or fast-food aesthetics."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No sans",
            "Muted natural color tones only",
        ],
    ),
    BenchmarkCase(
        case_id="case_14_tech_saas_platform",
        brief=(
            "Nexus is a B2B SaaS analytics platform targeting data teams at mid-size companies. "
            "The brand should feel precise, modern, and data-driven. "
            "Clean sans-serif fonts. Blues and grays preferred."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "Avoid serif fonts",
            "Blues and grays preferred",
        ],
    ),
    BenchmarkCase(
        case_id="case_15_luxury_hotel",
        brief=(
            "Serenova is a luxury boutique hotel brand targeting high-net-worth travelers aged 35-65. "
            "The brand should feel serene, opulent, and timeless. "
            "Soft neutrals and gold accents. No bright or casual colors."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No sans",
            "No bright or casual colors",
        ],
    ),
    BenchmarkCase(
        case_id="case_16_nonprofit_environment",
        brief=(
            "EarthKeep is an environmental nonprofit targeting donors and volunteers aged 20-50. "
            "The brand should feel hopeful, trustworthy, and mission-driven. "
            "Natural greens and blues. No corporate or profit-driven feel."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No red",
            "Natural greens and blues only",
        ],
    ),
    BenchmarkCase(
        case_id="case_17_wellness_app",
        brief=(
            "Calm Grove is a mental wellness and meditation app targeting stressed "
            "professionals aged 25-45. The brand should feel calming, safe, and nurturing. "
            "Soft muted tones only. No bold or aggressive aesthetics."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No red",
            "Soft muted tones only",
        ],
    ),
    BenchmarkCase(
        case_id="case_18_sustainable_fashion",
        brief=(
            "Thread & Earth is a sustainable fashion brand targeting eco-conscious "
            "millennials aged 22-35. The brand should feel ethical, modern, and aspirational. "
            "Earthy tones with a contemporary edge. No fast-fashion aesthetics."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No red",
            "Earthy tones required",
        ],
    ),
    BenchmarkCase(
        case_id="case_19_gaming_esports",
        brief=(
            "VoltX is an esports and gaming brand targeting competitive gamers aged 16-28. "
            "The brand should feel intense, futuristic, and electrifying. "
            "High contrast dark palette with neon accents welcome. Bold display fonts preferred."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "High contrast palette required",
            "Avoid serif fonts",
            "Dark background palette preferred",
        ],
    ),
    BenchmarkCase(
        case_id="case_20_premium_coffee",
        brief=(
            "Roast & Reserve is a premium single-origin coffee brand targeting "
            "specialty coffee enthusiasts aged 25-50. The brand should feel artisanal, "
            "warm, and intellectually curious. Rich earth tones. No corporate or mass-market feel."
        ),
        constraints=[
            "WCAG AA accessible palette",
            "Avoid neon colors",
            "No sans",
            "Rich earth tones required",
        ],
    ),
]


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
        if key and not os.environ.get(key):
            os.environ[key] = value


def get_openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for baseline benchmark.")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=key, base_url=base_url)


def safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def normalize_hex_list(colors: List[str]) -> List[str]:
    out: List[str] = []
    for c in colors or []:
        s = str(c).strip()
        if not s:
            continue
        if not s.startswith("#"):
            s = "#" + s
        if HEX_RE.match(s):
            out.append(s.upper())
    return out


def choose_font_pair(font_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not font_candidates:
        return {"headline": None, "body": None}
    headline = font_candidates[0]
    body = None
    for cand in font_candidates[1:]:
        if cand.get("category") != headline.get("category"):
            body = cand
            break
    if body is None:
        body = font_candidates[1] if len(font_candidates) > 1 else font_candidates[0]
    return {"headline": headline, "body": body}


def infer_archetype_heuristic(brief: str) -> str:
    text = brief.lower()
    mapping = [
        ("luxury", "Luxury"),
        ("jewelry", "Luxury"),
        ("fintech", "Corporate"),
        ("enterprise", "Corporate"),
        ("cybersecurity", "Tech"),
        ("saas", "Tech"),
        ("toy", "Playful"),
        ("kids", "Playful"),
        ("organic", "Organic"),
        ("eco", "Organic"),
        ("artisan", "Artisan"),
        ("bakery", "Artisan"),
        ("heritage", "Heritage"),
        ("minimal", "Minimal"),
        ("bold", "Bold"),
        ("youth", "Youthful"),
    ]
    for keyword, archetype in mapping:
        if keyword in text:
            return archetype
    return "Corporate"


def run_baseline_zero_shot(
    client: OpenAI,
    brief: str,
    constraints: List[str],
    model: str,
) -> Dict[str, Any]:
    prompt = f"""
You are a brand identity generator.
Given a brand brief and hard constraints, output one draft brand kit in a single pass.

Brand brief:
{brief}

Hard constraints:
{json.dumps(constraints, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "archetype": "<one of {ARCHETYPES}>",
  "font_recommendation": {{
    "headline": {{"family": "<font name>", "category": "<sans-serif|serif|display|handwriting|monospace>"}},
    "body": {{"family": "<font name>", "category": "<sans-serif|serif|display|handwriting|monospace>"}}
  }},
  "color_palette": {{
    "hex_codes": ["#RRGGBB", "#RRGGBB", "#RRGGBB", "#RRGGBB", "#RRGGBB"]
  }},
  "tone_keywords": ["k1", "k2", "k3"],
  "rationale": "<short explanation>"
}}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    parsed = safe_json_loads(resp.choices[0].message.content, {})
    archetype = parsed.get("archetype", "Corporate")
    if archetype not in ARCHETYPES:
        archetype = "Corporate"

    palette = normalize_hex_list((parsed.get("color_palette") or {}).get("hex_codes", []))
    kit = {
        "archetype": archetype,
        "font_recommendation": parsed.get("font_recommendation", {}),
        "color_palette": {"hex_codes": palette},
        "tone_keywords": parsed.get("tone_keywords", []),
        "rationale": parsed.get("rationale", ""),
    }
    return {"archetype": archetype, "kit": kit}


def run_baseline_rag_only(
    client: OpenAI,
    brief: str,
    constraints: List[str],
    model: str,
) -> Dict[str, Any]:
    prompt = f"""
Extract retrieval fields for brand identity generation from the brief.
Return only JSON:
{{
  "archetype": "<one of {ARCHETYPES}>",
  "industry": "<short>",
  "primary_emotions": ["e1","e2","e3"],
  "font_style": "<short style>",
  "style_keywords": ["k1","k2","k3"],
  "brand_attributes": ["a1","a2","a3"]
}}

Brief:
{brief}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    spec = safe_json_loads(resp.choices[0].message.content, {})
    archetype = spec.get("archetype", infer_archetype_heuristic(brief))
    if archetype not in ARCHETYPES:
        archetype = infer_archetype_heuristic(brief)

    font_candidates = font_lookup(archetype=archetype, style=spec.get("font_style", archetype), top_k=8)
    pair = choose_font_pair(font_candidates)

    emotions = spec.get("primary_emotions", [archetype.lower()])
    if not isinstance(emotions, list) or not emotions:
        emotions = [archetype.lower()]
    styles = spec.get("style_keywords", [archetype.lower()])
    if not isinstance(styles, list):
        styles = [archetype.lower()]

    palette_result = color_retrieve(
        emotions=[str(e).lower() for e in emotions[:3]],
        industry=str(spec.get("industry", "")),
        style_keywords=[str(s).lower() for s in styles[:5]],
        constraints=constraints,
        top_k=5,
    )
    hex_codes = normalize_hex_list((palette_result.get("best_palette") or {}).get("hex_codes", []))

    rules: List[str] = []
    attrs = spec.get("brand_attributes", [])
    if not isinstance(attrs, list):
        attrs = []
    for attr in attrs[:3]:
        for hit in heuristic_search(str(attr)):
            rule = str((hit or {}).get("rule", "")).strip()
            if rule and rule not in rules:
                rules.append(rule)

    kit = {
        "archetype": archetype,
        "font_recommendation": {
            "headline": pair["headline"],
            "body": pair["body"],
        },
        "color_palette": {"hex_codes": hex_codes},
        "design_rules": rules[:6],
        "rationale": "Single-pass retrieval baseline without QC loop.",
    }
    return {"archetype": archetype, "kit": kit}


def run_baseline_fontjoy(
    brief: str,
) -> Dict[str, Any]:
    archetype = infer_archetype_heuristic(brief)
    candidates = font_lookup(archetype=archetype, style="font pairing", top_k=8)
    pair = choose_font_pair(candidates)
    kit = {
        "archetype": archetype,
        "font_recommendation": {
            "headline": pair["headline"],
            "body": pair["body"],
        },
        "color_palette": {"hex_codes": []},
        "rationale": "Rule-based font pairing only baseline.",
    }
    return {"archetype": archetype, "kit": kit}


def extract_palette(kit: Dict[str, Any]) -> List[str]:
    return normalize_hex_list((kit.get("color_palette") or {}).get("hex_codes", []))


def extract_font_categories(kit: Dict[str, Any]) -> Tuple[str, str]:
    font_rec = kit.get("font_recommendation", {}) if isinstance(kit.get("font_recommendation"), dict) else {}
    head = font_rec.get("headline", {}) if isinstance(font_rec.get("headline"), dict) else {}
    body = font_rec.get("body", {}) if isinstance(font_rec.get("body"), dict) else {}
    return str(head.get("category", "")).lower(), str(body.get("category", "")).lower()


def hex_to_hsv(hex_code: str) -> Tuple[float, float, float]:
    code = hex_code.strip().lstrip("#")
    r, g, b = [int(code[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
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


def is_reddish(hex_code: str) -> bool:
    h, _, _ = hex_to_hsv(hex_code)
    return h < 20 or h > 340


def is_neon_like(hex_code: str) -> bool:
    _, s, v = hex_to_hsv(hex_code)
    return s >= 0.72 and v >= 0.82


def check_constraint(constraint: str, kit: Dict[str, Any], wcag_report: Dict[str, Any]) -> bool:
    text = constraint.lower().strip()
    palette = extract_palette(kit)
    head_cat, body_cat = extract_font_categories(kit)

    if any(k in text for k in ["wcag", "accessible", "accessibility", "contrast", "colorblind"]):
        return bool(palette) and bool(wcag_report.get("all_pass", False))

    if "no red" in text:
        return bool(palette) and not any(is_reddish(c) for c in palette)

    if "no neon" in text or "avoid neon" in text or "avoid harsh colors" in text:
        return bool(palette) and not any(is_neon_like(c) for c in palette)

    if "avoid serif" in text or "no serif" in text:
        return "serif" not in {head_cat, body_cat}

    if "avoid sans" in text or "no sans" in text:
        return "sans-serif" not in {head_cat, body_cat}

    return False


def score_coherence_llm(
    client: OpenAI,
    archetype: str,
    kit: Dict[str, Any],
    model: str,
) -> float:
    prompt = f"""
You are evaluating archetype coherence of a generated brand kit.
Archetype: {archetype}
Kit JSON:
{json.dumps(kit, ensure_ascii=False)}

Return only JSON:
{{
  "score": <float between 1.0 and 5.0>
}}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = safe_json_loads(resp.choices[0].message.content, {})
        score = float(parsed.get("score", 3.0))
    except Exception:
        score = 3.0
    return max(1.0, min(5.0, score))


def score_coherence_llm_rubric(
    client: OpenAI,
    archetype: str,
    kit: Dict[str, Any],
    model: str,
) -> float:
    prompt = f"""
You are a senior brand design critic. Score how coherently a generated brand
kit expresses its declared archetype on a 1.0-5.0 scale, using the rubric below.

Archetype: {archetype}
Kit JSON:
{json.dumps(kit, ensure_ascii=False)}

Score four sub-dimensions independently on 1.0-5.0 (use one decimal place,
do not bias toward integers):

1. font_alignment: Does the headline/body font category match the typographic
   conventions for this archetype (e.g. Luxury -> high-contrast serif or
   elegant display; Tech -> geometric sans; Artisan -> warm humanist serif)?
2. palette_alignment: Does the color palette evoke the archetype's emotional
   register (e.g. Luxury -> deep, restrained, gold/black; Playful -> bright,
   varied, high-saturation; Organic -> earthy, muted, low-chroma greens)?
3. tone_alignment: Are the tone_keywords / brand_attributes / palette_notes
   consistent with the archetype, with no contradictions?
4. narrative_coherence: Does the kit read as a unified story rather than a
   bag of independently retrieved parts? Is there explicit reasoning that
   ties choices to the archetype?

Anchors (apply to each sub-dimension):
- 5.0: textbook example of this archetype; experienced designer would ship as-is
- 4.0: clearly aligned; one minor element is generic but not contradictory
- 3.0: mixed; some elements fit, some are off; would need rework
- 2.0: mostly off-archetype; few elements match
- 1.0: contradicts the archetype

Reserve 5.0 for kits that are genuinely exemplary. If a kit has any minor flaw,
cap that sub-dimension at 4.5. If two or more sub-dimensions deserve 5.0,
double-check that nothing is generic boilerplate.

Then compute final_score as the unweighted average of the four sub-dimensions,
rounded to one decimal.

Return ONLY valid JSON in this exact schema:
{{
  "font_alignment": <float 1.0-5.0>,
  "palette_alignment": <float 1.0-5.0>,
  "tone_alignment": <float 1.0-5.0>,
  "narrative_coherence": <float 1.0-5.0>,
  "final_score": <float 1.0-5.0>,
  "rationale": "<one sentence justifying the lowest sub-dimension>"
}}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = safe_json_loads(resp.choices[0].message.content, {})
        score = float(parsed.get("final_score", parsed.get("score", 3.0)))
    except Exception:
        score = 3.0
    return max(1.0, min(5.0, score))


def evaluate_output(
    client: OpenAI,
    archetype: str,
    constraints: List[str],
    kit: Dict[str, Any],
    eval_model: str,
    wcag_mode: str = "min-pairs",
    wcag_min_pairs: int = 2,
    coherence_mode: str = "rubric",
) -> Dict[str, Any]:
    palette = extract_palette(kit)
    if wcag_mode == "min-pairs":
        wcag_report = evaluate_palette_wcag_min_pairs(
            palette, level="AA", large_text=False, min_pairs_required=wcag_min_pairs
        )
    else:
        wcag_report = evaluate_palette_wcag(palette, level="AA", large_text=False)

    constraint_results = []
    passed = 0
    for c in constraints:
        ok = check_constraint(c, kit, wcag_report)
        constraint_results.append({"constraint": c, "pass": ok})
        if ok:
            passed += 1

    if coherence_mode == "rubric":
        coherence = score_coherence_llm_rubric(
            client=client,
            archetype=archetype,
            kit=kit,
            model=eval_model,
        )
    else:
        coherence = score_coherence_llm(
            client=client,
            archetype=archetype,
            kit=kit,
            model=eval_model,
        )

    return {
        "wcag_all_pass": bool(wcag_report.get("all_pass", False)),
        "wcag_pass_rate": float(wcag_report.get("pass_rate", 0.0)),
        "coherence_score": round(coherence, 3),
        "constraint_pass_count": passed,
        "constraint_total": len(constraints),
        "constraint_results": constraint_results,
    }


def aggregate(system_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(system_runs)
    if n == 0:
        return {
            "cases": 0,
            "constraint_satisfaction_rate": 0.0,
            "wcag_pass_rate": 0.0,
            "archetype_coherence_score_mean": 0.0,
            "avg_runtime_sec": 0.0,
        }

    total_constraints = sum(r["evaluation"]["constraint_total"] for r in system_runs)
    total_constraint_pass = sum(r["evaluation"]["constraint_pass_count"] for r in system_runs)
    wcag_pass_count = sum(1 for r in system_runs if r["evaluation"]["wcag_all_pass"])
    coherence_scores = [r["evaluation"]["coherence_score"] for r in system_runs]
    runtimes = [r.get("runtime_sec", 0.0) for r in system_runs]

    return {
        "cases": n,
        "constraint_satisfaction_rate": round(
            (total_constraint_pass / total_constraints) if total_constraints else 0.0, 3
        ),
        "wcag_pass_rate": round(wcag_pass_count / n, 3),
        "archetype_coherence_score_mean": round(mean(coherence_scores), 3),
        "avg_runtime_sec": round(mean(runtimes), 3),
    }


def run_brandmind_full(brief: str, max_iterations: int) -> Dict[str, Any]:
    state = run_pipeline(brand_brief=brief, max_iterations=max_iterations)
    kit = state.get("approved_brand_kit") or state.get("draft_brand_kit") or {}
    archetype = state.get("archetype") or (kit.get("archetype") if isinstance(kit, dict) else "Corporate")
    if archetype not in ARCHETYPES:
        archetype = "Corporate"
    return {
        "archetype": archetype,
        "kit": kit if isinstance(kit, dict) else {},
        "state_status": state.get("status"),
        "iteration_count": state.get("iteration_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline benchmark vs BrandMind full model.")
    parser.add_argument("--out", type=str, default="reports/baseline_benchmark_report.json")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--zero-shot-model",
        type=str,
        default=os.environ.get("OPENAI_ZERO_SHOT_MODEL", "gpt-4o"),
    )
    parser.add_argument(
        "--rag-model",
        type=str,
        default=os.environ.get("OPENAI_RAG_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--wcag-mode",
        type=str,
        default="min-pairs",
        choices=["legacy", "min-pairs"],
    )
    parser.add_argument("--wcag-min-pairs", type=int, default=2)
    parser.add_argument(
        "--coherence-mode",
        type=str,
        default="rubric",
        choices=["legacy", "rubric"],
    )
    args = parser.parse_args()

    load_dotenv_file(Path(".env"))
    client = get_openai_client()

    systems = ["brandmind_full", "baseline_zero_shot_gpt4o", "baseline_rag_only", "baseline_fontjoy"]
    all_runs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in systems}

    for case in BENCHMARK_CASES:
        print(f"\n=== Running benchmark case: {case.case_id} ===")
        for system in systems:
            t0 = time.time()
            if system == "brandmind_full":
                output = run_brandmind_full(case.brief, max_iterations=args.max_iterations)
            elif system == "baseline_zero_shot_gpt4o":
                output = run_baseline_zero_shot(
                    client=client,
                    brief=case.brief,
                    constraints=case.constraints,
                    model=args.zero_shot_model,
                )
            elif system == "baseline_rag_only":
                output = run_baseline_rag_only(
                    client=client,
                    brief=case.brief,
                    constraints=case.constraints,
                    model=args.rag_model,
                )
            else:
                output = run_baseline_fontjoy(case.brief)

            eval_report = evaluate_output(
                client=client,
                archetype=output.get("archetype", "Corporate"),
                constraints=case.constraints,
                kit=output.get("kit", {}),
                eval_model=args.eval_model,
                wcag_mode=args.wcag_mode,
                wcag_min_pairs=args.wcag_min_pairs,
                coherence_mode=args.coherence_mode,
            )
            elapsed = time.time() - t0
            row = {
                "case_id": case.case_id,
                "system": system,
                "brief": case.brief,
                "constraints": case.constraints,
                "runtime_sec": round(elapsed, 3),
                "output": output,
                "evaluation": eval_report,
            }
            all_runs[system].append(row)
            print(
                f"[{system}] coherence={eval_report['coherence_score']} "
                f"constraint={eval_report['constraint_pass_count']}/{eval_report['constraint_total']} "
                f"wcag_case_pass={eval_report['wcag_all_pass']} runtime={round(elapsed,2)}s"
            )

    summary = {system: aggregate(rows) for system, rows in all_runs.items()}
    ranking = sorted(
        summary.items(),
        key=lambda kv: (
            kv[1]["constraint_satisfaction_rate"],
            kv[1]["wcag_pass_rate"],
            kv[1]["archetype_coherence_score_mean"],
        ),
        reverse=True,
    )

    payload = {
        "cases_count": len(BENCHMARK_CASES),
        "summary": summary,
        "ranking_by_metrics": [{"system": k, **v} for k, v in ranking],
        "details": all_runs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())