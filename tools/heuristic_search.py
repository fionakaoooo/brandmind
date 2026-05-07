%%writefile /content/brandmind/tools/heuristic_search.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

LEARNING_RATE = 0.12
BASELINE_SCORE = 0.60
MIN_WEIGHT = 0.10
MAX_WEIGHT = 3.00


_RULE_BANK: Dict[str, List[Dict[str, Any]]] = {

    "premium": [
        {"rule": "Use restrained color contrast and generous whitespace.", "id": "premium_0"},
        {"rule": "Prefer elegant typography with a refined visual hierarchy.", "id": "premium_1"},
        {"rule": "Avoid loud or high-saturation colors that undermine premium tone.", "id": "premium_2"},
    ],
    "refined": [
        {"rule": "Reduce visual clutter and keep layouts highly ordered.", "id": "refined_0"},
        {"rule": "Use subtle accents instead of loud, high-saturation highlights.", "id": "refined_1"},
        {"rule": "Maintain generous margins and let negative space carry the composition.", "id": "refined_2"},
    ],
    "modern": [
        {"rule": "Favor clean alignment, simple geometry, and minimal ornament.", "id": "modern_0"},
        {"rule": "Use contemporary sans or high-contrast serif pairings.", "id": "modern_1"},
        {"rule": "Avoid decorative or historical typographic references.", "id": "modern_2"},
    ],
    "trustworthy": [
        {"rule": "Maintain strong readability and consistent spacing throughout.", "id": "trustworthy_0"},
        {"rule": "Favor balanced compositions and stable, calm visual rhythm.", "id": "trustworthy_1"},
        {"rule": "Use conservative color anchors such as deep navy or neutral grey.", "id": "trustworthy_2"},
    ],
    "soft": [
        {"rule": "Use gentle tonal transitions and avoid harsh contrast jumps.", "id": "soft_0"},
        {"rule": "Favor rounded or graceful forms and a calm palette.", "id": "soft_1"},
        {"rule": "Keep saturation low across the entire palette.", "id": "soft_2"},
    ],
    "playful": [
        {"rule": "Allow brighter accents and more energetic composition.", "id": "playful_0"},
        {"rule": "Use friendlier typography and slightly more visual motion.", "id": "playful_1"},
        {"rule": "Introduce unexpected color pops against a neutral ground.", "id": "playful_2"},
    ],
    "organic": [
        {"rule": "Favor earthy or natural hues and softer saturation.", "id": "organic_0"},
        {"rule": "Use human, tactile typography and less rigid composition.", "id": "organic_1"},
        {"rule": "Avoid synthetic-looking gradients or overly geometric forms.", "id": "organic_2"},
    ],
    "bold": [
        {"rule": "Use stronger contrast and a more assertive typographic scale.", "id": "bold_0"},
        {"rule": "Favor impactful focal points and simplified messaging.", "id": "bold_1"},
        {"rule": "Use high-saturation accent colors sparingly for maximum impact.", "id": "bold_2"},
    ],
    "classic": [
        {"rule": "Use timeless typography and balanced traditional proportions.", "id": "classic_0"},
        {"rule": "Avoid trend-heavy decorative choices.", "id": "classic_1"},
        {"rule": "Reference established typographic conventions without literal pastiche.", "id": "classic_2"},
    ],
    "minimal": [
        {"rule": "Maximise negative space and resist adding decorative detail.", "id": "minimal_0"},
        {"rule": "Limit the active palette to two or three carefully chosen colours.", "id": "minimal_1"},
        {"rule": "Use a single typeface family at two weights only.", "id": "minimal_2"},
    ],
    "luxury": [
        {"rule": "Use deep tones and metallic or neutral accent highlights.", "id": "luxury_0"},
        {"rule": "Prefer high-contrast type pairings with fine weight variation.", "id": "luxury_1"},
        {"rule": "Maximise whitespace; scarcity of elements signals scarcity of product.", "id": "luxury_2"},
    ],
    "tech": [
        {"rule": "Use cool, high-contrast colours with strong geometric forms.", "id": "tech_0"},
        {"rule": "Prefer monospaced or geometric sans fonts for data contexts.", "id": "tech_1"},
        {"rule": "Maintain strict grid alignment and pixel-precise spacing.", "id": "tech_2"},
    ],
    "artisan": [
        {"rule": "Lean into texture cues: use fonts that feel hand-crafted.", "id": "artisan_0"},
        {"rule": "Muted, natural tones signal authenticity and craft.", "id": "artisan_1"},
        {"rule": "Avoid overly polished or symmetrical layouts; slight irregularity signals humanity.", "id": "artisan_2"},
    ],
    "heritage": [
        {"rule": "Reference historical typographic forms without pastiche.", "id": "heritage_0"},
        {"rule": "Use sepia, navy, or aged-paper tones for a sense of time.", "id": "heritage_1"},
        {"rule": "Incorporate classical proportions and established serif traditions.", "id": "heritage_2"},
    ],
    "youthful": [
        {"rule": "Use vibrant accent colours offset by clean, open space.", "id": "youthful_0"},
        {"rule": "Favour rounded, friendly display typefaces over rigid geometry.", "id": "youthful_1"},
        {"rule": "Keep layouts dynamic and asymmetric to signal energy.", "id": "youthful_2"},
    ],
    "corporate": [
        {"rule": "Prioritise legibility and ordered grid-based layouts.", "id": "corporate_0"},
        {"rule": "Use conservative blues or greys to project stability.", "id": "corporate_1"},
        {"rule": "Maintain strict typographic hierarchy across all touchpoints.", "id": "corporate_2"},
    ],
    "eco_friendly": [
        {"rule": "Use muted, natural greens and earth tones; avoid synthetic-looking hues.", "id": "eco_friendly_0"},
        {"rule": "Favour organic shapes and textures that signal handmade or natural origin.", "id": "eco_friendly_1"},
        {"rule": "Avoid neon or highly saturated colours that contradict an environmental ethos.", "id": "eco_friendly_2"},
    ],
    "sustainable": [
        {"rule": "Limit the palette to a small set of low-saturation, nature-derived colours.", "id": "sustainable_0"},
        {"rule": "Choose typography that feels considered and purposeful, not decorative.", "id": "sustainable_1"},
        {"rule": "Use restrained layouts that avoid visual waste or unnecessary embellishment.", "id": "sustainable_2"},
    ],
    "innovative": [
        {"rule": "Use asymmetric layouts and unexpected typographic scale contrasts.", "id": "innovative_0"},
        {"rule": "Pair a geometric sans with a high-contrast display face to signal forward thinking.", "id": "innovative_1"},
        {"rule": "Introduce one unexpected accent colour against an otherwise neutral palette.", "id": "innovative_2"},
    ],
    "reliable": [
        {"rule": "Anchor layouts with strong horizontal structure and consistent grid alignment.", "id": "reliable_0"},
        {"rule": "Use deep navy, charcoal, or forest green as primary tones to project dependability.", "id": "reliable_1"},
        {"rule": "Avoid decorative typefaces; stick to proven, widely-readable families.", "id": "reliable_2"},
    ],
    "elegant": [
        {"rule": "Maximise negative space and use type size contrast instead of decorative elements.", "id": "elegant_0"},
        {"rule": "Restrict the palette to two tones with one metallic or neutral accent.", "id": "elegant_1"},
        {"rule": "Use fine-weight serif or tall sans-serif faces for headline text.", "id": "elegant_2"},
    ],
    "warm": [
        {"rule": "Use amber, terracotta, or soft ochre as dominant tones with cream as a ground.", "id": "warm_0"},
        {"rule": "Favour rounded letterforms and generous line-height for an inviting feel.", "id": "warm_1"},
        {"rule": "Avoid cool greys or stark white backgrounds that undercut warmth.", "id": "warm_2"},
    ],
    "clean": [
        {"rule": "Maintain strict whitespace discipline; every element must have a clear purpose.", "id": "clean_0"},
        {"rule": "Use a single sans-serif family at two weights only — no decorative extras.", "id": "clean_1"},
        {"rule": "Align all elements to a consistent grid; avoid floating or orphaned components.", "id": "clean_2"},
    ],
    "professional": [
        {"rule": "Use structured, grid-based layouts with consistent margin and padding ratios.", "id": "professional_0"},
        {"rule": "Restrict typefaces to established, neutral families with strong legibility.", "id": "professional_1"},
        {"rule": "Avoid playful colour choices; neutral palettes with one accent are safest.", "id": "professional_2"},
    ],
    "energetic": [
        {"rule": "Use high-contrast colour pairings with a saturated accent to create visual momentum.", "id": "energetic_0"},
        {"rule": "Apply diagonal or dynamic compositional lines to suggest movement.", "id": "energetic_1"},
        {"rule": "Use bold typographic weight contrasts to inject visual energy.", "id": "energetic_2"},
    ],
    "calming": [
        {"rule": "Use low-saturation cool tones — muted blues, sage greens, or soft lavenders.", "id": "calming_0"},
        {"rule": "Avoid tight kerning and dense layouts; let the design breathe.", "id": "calming_1"},
        {"rule": "Use generous line-height and ample paragraph spacing throughout.", "id": "calming_2"},
    ],
    "authentic": [
        {"rule": "Avoid overly polished or symmetrical layouts; slight irregularity signals humanity.", "id": "authentic_0"},
        {"rule": "Use typefaces with visible stroke variation or hand-crafted quality.", "id": "authentic_1"},
        {"rule": "Let natural imperfection show — perfect symmetry can undermine authenticity.", "id": "authentic_2"},
    ],
    "grounded": [
        {"rule": "Anchor the palette with a dark earth tone as the primary background or headline colour.", "id": "grounded_0"},
        {"rule": "Use stable, horizontal compositions that avoid excessive visual tension.", "id": "grounded_1"},
        {"rule": "Favour serif or slab-serif typefaces that feel rooted and substantial.", "id": "grounded_2"},
    ],
    "sophisticated": [
        {"rule": "Use high-contrast type hierarchies with generous tracking on display text.", "id": "sophisticated_0"},
        {"rule": "Restrict accent colours to one — let form and proportion carry the weight.", "id": "sophisticated_1"},
        {"rule": "Use fine typographic details: small caps, ligatures, or optical sizing where available.", "id": "sophisticated_2"},
    ],
    "natural": [
        {"rule": "Draw from a palette of forest, clay, stone, and water-inspired hues.", "id": "natural_0"},
        {"rule": "Use serif or humanist sans typefaces that reference pre-digital craft traditions.", "id": "natural_1"},
        {"rule": "Incorporate irregular organic shapes rather than strict geometric forms.", "id": "natural_2"},
    ],
    "precise": [
        {"rule": "Use monospaced or geometric typefaces to signal exactness and technical rigour.", "id": "precise_0"},
        {"rule": "Maintain tight, consistent spacing and avoid any decorative excess.", "id": "precise_1"},
        {"rule": "Use a strictly limited colour palette — no more than three functional colours.", "id": "precise_2"},
    ],
    "timeless": [
        {"rule": "Avoid trend-driven typefaces; choose forms with 50+ years of proven readability.", "id": "timeless_0"},
        {"rule": "Use neutral, desaturated palettes that do not reference a specific decade.", "id": "timeless_1"},
        {"rule": "Rely on classical proportions and established compositional rules.", "id": "timeless_2"},
    ],
    "exclusive": [
        {"rule": "Use a very limited palette — black, white, and one precious-metal accent only.", "id": "exclusive_0"},
        {"rule": "Maximise white space; scarcity of elements signals scarcity of product.", "id": "exclusive_1"},
        {"rule": "Use fine-weight display type with wide tracking to signal rarity.", "id": "exclusive_2"},
    ],
    "friendly": [
        {"rule": "Use rounded display typefaces and avoid sharp angular letterforms.", "id": "friendly_0"},
        {"rule": "Choose warm mid-tone palettes — peach, sky blue, soft yellow — over cold neutrals.", "id": "friendly_1"},
        {"rule": "Keep layouts open and approachable; avoid dense or intimidating compositions.", "id": "friendly_2"},
    ],
    "efficient": [
        {"rule": "Strip layouts to the minimum viable elements; remove anything that does not inform.", "id": "efficient_0"},
        {"rule": "Use a tight type scale with clear hierarchy — readers should never have to search.", "id": "efficient_1"},
        {"rule": "Prefer functional sans-serif typefaces over expressive or decorative alternatives.", "id": "efficient_2"},
    ],
    "authoritative": [
        {"rule": "Use a strong, high-contrast serif or condensed sans as the primary display face.", "id": "authoritative_0"},
        {"rule": "Anchor the palette with a deep, saturated primary colour — navy, forest, or burgundy.", "id": "authoritative_1"},
        {"rule": "Use large typographic scale contrasts to project confidence and command.", "id": "authoritative_2"},
    ],
    "cutting_edge": [
        {"rule": "Use stark, high-contrast colour pairings with a single neon or electric accent.", "id": "cutting_edge_0"},
        {"rule": "Favour experimental typefaces or extreme weight contrasts within one family.", "id": "cutting_edge_1"},
        {"rule": "Break grid conventions deliberately to signal that conventions do not apply.", "id": "cutting_edge_2"},
    ],
    "secure": [
        {"rule": "Use deep, dark backgrounds with high-contrast light text to project control.", "id": "secure_0"},
        {"rule": "Favour geometric, structured sans-serif typefaces over humanist alternatives.", "id": "secure_1"},
        {"rule": "Avoid warm or playful tones; cool blues and greys signal protection and stability.", "id": "secure_2"},
    ],
    "technical": [
        {"rule": "Use monospaced typefaces for data display and code-adjacent contexts.", "id": "technical_0"},
        {"rule": "Maintain strict grid alignment and avoid organic or freeform compositional choices.", "id": "technical_1"},
        {"rule": "Use cool, desaturated palettes with one high-visibility accent for key actions.", "id": "technical_2"},
    ],
    "handmade": [
        {"rule": "Use typefaces with visible ink-trap details or hand-drawn qualities.", "id": "handmade_0"},
        {"rule": "Introduce deliberate texture and slight irregularity into layout and colour application.", "id": "handmade_1"},
        {"rule": "Use warm, slightly impure colour mixes rather than digitally pure hex values.", "id": "handmade_2"},
    ],
    "neighborly": [
        {"rule": "Use approachable, mid-weight sans or slab-serif typefaces that feel unpretentious.", "id": "neighborly_0"},
        {"rule": "Choose warm, familiar colour palettes — terracotta, cream, soft green.", "id": "neighborly_1"},
        {"rule": "Keep layouts informal and conversational; avoid corporate rigidity.", "id": "neighborly_2"},
    ],
    "craft_driven": [
        {"rule": "Use typography that shows evidence of the designer's hand — slight optical corrections.", "id": "craft_driven_0"},
        {"rule": "Draw the palette from natural, craft-associated materials: wood, linen, clay, ink.", "id": "craft_driven_1"},
        {"rule": "Allow asymmetry and visual weight imbalance to signal deliberate craft decision-making.", "id": "craft_driven_2"},
    ],
    "established": [
        {"rule": "Use classic serif typefaces with long publication histories.", "id": "established_0"},
        {"rule": "Anchor layouts in traditional proportions and proven grid systems.", "id": "established_1"},
        {"rule": "Use muted, historically-grounded colour palettes that avoid trend associations.", "id": "established_2"},
    ],
    "intentional": [
        {"rule": "Every visual element must have an explicit reason for being present.", "id": "intentional_0"},
        {"rule": "Use restraint as the primary design tool — say more with less.", "id": "intentional_1"},
        {"rule": "Align typeface choices directly with the brand's core values, not trend.", "id": "intentional_2"},
    ],
    "lighthearted": [
        {"rule": "Use soft pastel palettes with one brighter accent for contrast.", "id": "lighthearted_0"},
        {"rule": "Favour rounded, display typefaces that smile rather than shout.", "id": "lighthearted_1"},
        {"rule": "Keep layouts airy and uncluttered to maintain a sense of ease.", "id": "lighthearted_2"},
    ],
    "spontaneous": [
        {"rule": "Break strict grid alignment occasionally to suggest energy and immediacy.", "id": "spontaneous_0"},
        {"rule": "Use saturated, unexpected colour combinations that feel instinctive not calculated.", "id": "spontaneous_1"},
        {"rule": "Introduce expressive typographic variation — scale, weight, rotation — within a clear system.", "id": "spontaneous_2"},
    ],
}


def _default_weights() -> Dict[str, float]:
    return {
        rule["id"]: 1.0
        for rules in _RULE_BANK.values()
        for rule in rules
    }


def heuristic_search(
    brand_attribute: str,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    attr = (brand_attribute or "").strip().lower().replace("-", "_").replace(" ", "_")
    w = weights or {}

    candidates = _RULE_BANK.get(attr)

    if candidates is None:
        fallback = [
            {
                "rule": f"Use layout, typography, and color choices to reinforce {attr}.",
                "id": f"fallback_{attr}_0",
            },
            {
                "rule": f"Keep the overall visual system consistent with the attribute '{attr}'.",
                "id": f"fallback_{attr}_1",
            },
        ]
        enriched = [{**r, "weight": w.get(r["id"], 1.0)} for r in fallback]
        return sorted(enriched, key=lambda r: r["weight"], reverse=True)

    enriched = [{**rule, "weight": w.get(rule["id"], 1.0)} for rule in candidates]
    return sorted(enriched, key=lambda r: r["weight"], reverse=True)


def initialise_weights(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("heuristic_weights"):
        state = {**state, "heuristic_weights": _default_weights()}
    return state


def _extract_used_rule_ids(state: Dict[str, Any]) -> List[str]:
    explicit_ids = state.get("used_heuristic_rule_ids") or []
    if explicit_ids:
        return [str(x) for x in explicit_ids]

    kit = state.get("draft_brand_kit") or {}
    design_rules: List[str] = kit.get("design_rules", [])
    if not design_rules:
        return []

    text_to_id: Dict[str, str] = {
        rule["rule"]: rule["id"]
        for rules in _RULE_BANK.values()
        for rule in rules
    }
    return [text_to_id[r] for r in design_rules if r in text_to_id]


def _compute_feedback_score(qc_scores: Dict[str, Any]) -> float:
    if not qc_scores:
        return BASELINE_SCORE

    wcag_pass_rate       = float(qc_scores.get("wcag", {}).get("pass_rate", 0.0))
    coherence_score      = float(qc_scores.get("coherence", {}).get("score", 0.0)) / 5.0
    constraint_pass_rate = float(qc_scores.get("constraints", {}).get("pass_rate", 0.0))

    feedback_score = (
        0.40 * wcag_pass_rate
      + 0.35 * coherence_score
      + 0.25 * constraint_pass_rate
    )
    return max(0.0, min(1.0, round(feedback_score, 4)))


def update_heuristic_weights(state: Dict[str, Any]) -> Dict[str, Any]:
    weights: Dict[str, float] = dict(state.get("heuristic_weights") or _default_weights())
    used_ids = _extract_used_rule_ids(state)
    qc_scores = state.get("qc_scores") or {}

    if not used_ids:
        print("[HeuristicSearch] No used heuristic rules found in draft_brand_kit.")
        return {**state, "heuristic_weights": weights}

    feedback_score = _compute_feedback_score(qc_scores)
    delta = LEARNING_RATE * (feedback_score - BASELINE_SCORE)

    for rule_id in used_ids:
        old = weights.get(rule_id, 1.0)
        new = old + delta
        new = max(MIN_WEIGHT, min(MAX_WEIGHT, new))
        weights[rule_id] = round(new, 4)

    print(
        f"[HeuristicSearch] Continuous update applied to {len(used_ids)} rule(s). "
        f"feedback_score={feedback_score:.4f}, delta={delta:+.4f}"
    )
    return {**state, "heuristic_weights": weights}


def get_top_rules(weights: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
    all_rules = [
        {
            "attribute": attr,
            "rule": rule["rule"],
            "id": rule["id"],
            "weight": weights.get(rule["id"], 1.0),
        }
        for attr, rules in _RULE_BANK.items()
        for rule in rules
    ]
    return sorted(all_rules, key=lambda x: x["weight"], reverse=True)[:top_k]


def inspect_attribute_rules(
    attribute: str,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    return heuristic_search(attribute, weights=weights)


if __name__ == "__main__":
    mock_state: Dict[str, Any] = {
        "heuristic_weights": None,
        "draft_brand_kit": {
            "design_rules": [
                "Use restrained color contrast and generous whitespace.",
                "Favor earthy or natural hues and softer saturation.",
            ]
        },
        "qc_scores": {
            "wcag": {"pass_rate": 0.60},
            "coherence": {"score": 3.5},
            "constraints": {"pass_rate": 0.67},
        },
    }

    mock_state = initialise_weights(mock_state)
    print("Total rules:", sum(len(v) for v in _RULE_BANK.values()))
    print("Total attributes:", len(_RULE_BANK))

    mock_state = update_heuristic_weights(mock_state)

    print("\nUpdated premium rules:")
    for r in inspect_attribute_rules("premium", weights=mock_state["heuristic_weights"]):
        print(f"  [{r['weight']:.4f}] {r['id']} -> {r['rule']}")

    print("\nTop 5 rules overall:")
    for entry in get_top_rules(mock_state["heuristic_weights"], top_k=5):
        print(f"  [{entry['weight']:.4f}] ({entry['attribute']}) {entry['rule']}")
