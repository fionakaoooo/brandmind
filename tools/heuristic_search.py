from __future__ import annotations

from typing import Any, Dict, List, Optional


LEARNING_RATE = 0.12
BASELINE_SCORE = 0.60
MIN_WEIGHT = 0.10
MAX_WEIGHT = 3.00


ATTRIBUTE_ALIASES: Dict[str, str] = {
    # sustainability / organic
    "eco-friendly": "sustainable",
    "eco_friendly": "sustainable",
    "eco": "sustainable",
    "green": "sustainable",
    "environmental": "sustainable",
    "earthly": "natural",
    "earthy": "natural",
    "botanical": "natural",

    # precision / tech
    "accurate": "precise",
    "accuracy": "precise",
    "exact": "precise",
    "technical": "tech",
    "engineering": "technical",
    "advanced": "innovative",
    "forward-thinking": "innovative",
    "forward_thinking": "innovative",
    "futuristic": "cutting_edge",
    "cutting-edge": "cutting_edge",

    # professional / corporate
    "finance": "corporate",
    "financial": "corporate",
    "enterprise": "corporate",
    "b2b": "corporate",
    "user-centric": "professional",
    "user_centric": "professional",
    "business": "professional",
    "stable": "trustworthy",
    "secure": "secure",
    "security": "secure",

    # playful / youthful
    "imaginative": "playful",
    "fun": "playful",
    "kid-friendly": "playful",
    "kid_friendly": "playful",
    "child-friendly": "playful",
    "child_friendly": "playful",
    "approachable": "friendly",

    # luxury / heritage
    "exclusivity": "exclusive",
    "exclusive": "exclusive",
    "high-end": "luxury",
    "high_end": "luxury",
    "heirloom": "heritage",
    "craftsmanship": "craft_driven",
    "craft": "craft_driven",
    "craft-driven": "craft_driven",

    # artisan
    "hand-crafted": "handmade",
    "handcrafted": "handmade",
    "human": "authentic",
    "human-made": "handmade",
    "human_made": "handmade",
    "neighborhoody": "neighborly",
    "neighborhood": "neighborly",

    # tone cleanup
    "calm": "calming",
    "warmth": "warm",
    "inviting": "friendly",
    "welcoming": "friendly",

    "trust":                "trustworthy",
    "precision":            "precise",
    "modernity":            "modern",
    "high-energy":          "energetic",
    "high_energy":          "energetic",
    "rebellious":           "bold",
    "fearless":             "bold",
    "expressive":           "bold",
    "artisanal":            "artisan",
    "community-focused":    "neighborly",
    "community_focused":    "neighborly",
    "nostalgic":            "heritage",
    "sophistication":       "sophisticated",
    "pure":                 "clean",
    "engaging":             "playful",
    "accessible":           "clean",
    "reliable":             "trustworthy",
    "dynamic":              "energetic",
}


# ─────────────────────────────────────────────────────────────────────────────
# Rule bank
# 每个 key 对应一组可解释的 design rules
# ─────────────────────────────────────────────────────────────────────────────

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
        {"rule": "Use contemporary sans-serif or high-contrast serif pairings.", "id": "modern_1"},
        {"rule": "Avoid decorative or historical typographic references unless required by the archetype.", "id": "modern_2"},
    ],

    "trustworthy": [
        {"rule": "Maintain strong readability and consistent spacing throughout.", "id": "trustworthy_0"},
        {"rule": "Favor balanced compositions and a stable, calm visual rhythm.", "id": "trustworthy_1"},
        {"rule": "Use conservative anchors such as deep navy, charcoal, forest green, or neutral grey.", "id": "trustworthy_2"},
    ],

    "professional": [
        {"rule": "Use structured, grid-based layouts with consistent margin and padding ratios.", "id": "professional_0"},
        {"rule": "Restrict typefaces to established, neutral families with strong legibility.", "id": "professional_1"},
        {"rule": "Avoid playful color choices; neutral palettes with one controlled accent are safest.", "id": "professional_2"},
    ],

    "corporate": [
        {"rule": "Prioritize legibility and ordered grid-based layouts.", "id": "corporate_0"},
        {"rule": "Use conservative blues, greys, or neutral anchors to project stability.", "id": "corporate_1"},
        {"rule": "Maintain strict typographic hierarchy across all touchpoints.", "id": "corporate_2"},
    ],

    "secure": [
        {"rule": "Use deep, dark backgrounds with high-contrast light text to project control.", "id": "secure_0"},
        {"rule": "Favor geometric, structured sans-serif typefaces over decorative alternatives.", "id": "secure_1"},
        {"rule": "Avoid warm or playful tones; cool blues and greys signal protection and stability.", "id": "secure_2"},
    ],

    "precise": [
        {"rule": "Use monospaced or geometric typefaces to signal exactness and technical rigor.", "id": "precise_0"},
        {"rule": "Maintain tight, consistent spacing and avoid decorative excess.", "id": "precise_1"},
        {"rule": "Use a strictly limited color palette with clearly assigned functional roles.", "id": "precise_2"},
    ],

    "efficient": [
        {"rule": "Strip layouts to the minimum viable elements; remove anything that does not inform.", "id": "efficient_0"},
        {"rule": "Use a tight type scale with clear hierarchy so readers never have to search.", "id": "efficient_1"},
        {"rule": "Prefer functional sans-serif typefaces over expressive or decorative alternatives.", "id": "efficient_2"},
    ],

    "tech": [
        {"rule": "Use cool, high-contrast colors with strong geometric forms.", "id": "tech_0"},
        {"rule": "Prefer monospaced or geometric sans fonts for technical or data contexts.", "id": "tech_1"},
        {"rule": "Maintain strict grid alignment and pixel-precise spacing.", "id": "tech_2"},
    ],

    "technical": [
        {"rule": "Use monospaced typefaces for data display and code-adjacent contexts.", "id": "technical_0"},
        {"rule": "Maintain strict grid alignment and avoid organic or freeform compositional choices.", "id": "technical_1"},
        {"rule": "Use cool, desaturated palettes with one high-visibility accent for key actions.", "id": "technical_2"},
    ],

    "innovative": [
        {"rule": "Use asymmetric layouts and unexpected typographic scale contrasts.", "id": "innovative_0"},
        {"rule": "Pair a geometric sans with a high-contrast display face to signal forward thinking.", "id": "innovative_1"},
        {"rule": "Introduce one unexpected accent color against an otherwise neutral palette.", "id": "innovative_2"},
    ],

    "cutting_edge": [
        {"rule": "Use stark, high-contrast color pairings with a single electric accent.", "id": "cutting_edge_0"},
        {"rule": "Favor experimental type hierarchy or extreme weight contrasts within one family.", "id": "cutting_edge_1"},
        {"rule": "Break grid conventions deliberately but keep the system readable.", "id": "cutting_edge_2"},
    ],

    "minimal": [
        {"rule": "Maximize negative space and resist adding decorative detail.", "id": "minimal_0"},
        {"rule": "Limit the active palette to two or three carefully chosen colors.", "id": "minimal_1"},
        {"rule": "Use a single typeface family at two weights when possible.", "id": "minimal_2"},
    ],

    "clean": [
        {"rule": "Maintain strict whitespace discipline; every element must have a clear purpose.", "id": "clean_0"},
        {"rule": "Use simple type hierarchy with no decorative extras.", "id": "clean_1"},
        {"rule": "Align all elements to a consistent grid; avoid floating or orphaned components.", "id": "clean_2"},
    ],

    "organic": [
        {"rule": "Favor earthy or natural hues and softer saturation.", "id": "organic_0"},
        {"rule": "Use human, tactile typography and less rigid composition.", "id": "organic_1"},
        {"rule": "Avoid synthetic-looking gradients or overly geometric forms.", "id": "organic_2"},
    ],

    "natural": [
        {"rule": "Draw from forest, clay, stone, linen, and water-inspired hues.", "id": "natural_0"},
        {"rule": "Use serif or humanist sans typefaces that reference pre-digital craft traditions.", "id": "natural_1"},
        {"rule": "Incorporate irregular organic shapes rather than strict geometric forms.", "id": "natural_2"},
    ],

    "sustainable": [
        {"rule": "Limit the palette to a small set of low-saturation, nature-derived colors.", "id": "sustainable_0"},
        {"rule": "Choose typography that feels considered and purposeful, not decorative.", "id": "sustainable_1"},
        {"rule": "Use restrained layouts that avoid visual waste or unnecessary embellishment.", "id": "sustainable_2"},
    ],

    "eco_friendly": [
        {"rule": "Use muted greens and earth tones; avoid synthetic-looking hues.", "id": "eco_friendly_0"},
        {"rule": "Favor organic shapes and textures that signal natural origin.", "id": "eco_friendly_1"},
        {"rule": "Avoid neon or highly saturated colors that contradict an environmental ethos.", "id": "eco_friendly_2"},
    ],

    "grounded": [
        {"rule": "Anchor the palette with a dark earth tone as the primary background or headline color.", "id": "grounded_0"},
        {"rule": "Use stable, horizontal compositions that avoid excessive visual tension.", "id": "grounded_1"},
        {"rule": "Favor serif or slab-serif typefaces that feel rooted and substantial.", "id": "grounded_2"},
    ],

    "calming": [
        {"rule": "Use low-saturation cool tones such as muted blues, sage greens, or soft lavenders.", "id": "calming_0"},
        {"rule": "Avoid tight kerning and dense layouts; let the design breathe.", "id": "calming_1"},
        {"rule": "Use generous line-height and ample paragraph spacing throughout.", "id": "calming_2"},
    ],

    "soft": [
        {"rule": "Use gentle tonal transitions and avoid harsh contrast jumps.", "id": "soft_0"},
        {"rule": "Favor rounded or graceful forms and a calm palette.", "id": "soft_1"},
        {"rule": "Keep saturation low across the entire palette.", "id": "soft_2"},
    ],

    "warm": [
        {"rule": "Use amber, terracotta, soft ochre, or cream to create an inviting feel.", "id": "warm_0"},
        {"rule": "Favor rounded letterforms and generous line-height.", "id": "warm_1"},
        {"rule": "Avoid cold greys or stark white backgrounds that undercut warmth.", "id": "warm_2"},
    ],

    "friendly": [
        {"rule": "Use rounded display typefaces and avoid sharp angular letterforms.", "id": "friendly_0"},
        {"rule": "Choose warm mid-tone palettes over cold corporate neutrals.", "id": "friendly_1"},
        {"rule": "Keep layouts open and approachable; avoid dense or intimidating compositions.", "id": "friendly_2"},
    ],

    "playful": [
        {"rule": "Allow brighter accents and more energetic composition.", "id": "playful_0"},
        {"rule": "Use friendlier typography and slightly more visual motion.", "id": "playful_1"},
        {"rule": "Introduce unexpected color pops against a neutral ground.", "id": "playful_2"},
    ],

    "youthful": [
        {"rule": "Use vibrant accent colors offset by clean, open space.", "id": "youthful_0"},
        {"rule": "Favor rounded, friendly display typefaces over rigid geometry.", "id": "youthful_1"},
        {"rule": "Keep layouts dynamic and asymmetric to signal energy.", "id": "youthful_2"},
    ],

    "energetic": [
        {"rule": "Use high-contrast color pairings with a saturated accent to create visual momentum.", "id": "energetic_0"},
        {"rule": "Apply diagonal or dynamic compositional lines to suggest movement.", "id": "energetic_1"},
        {"rule": "Use bold typographic weight contrasts to inject visual energy.", "id": "energetic_2"},
    ],

    "lighthearted": [
        {"rule": "Use soft pastel palettes with one brighter accent for contrast.", "id": "lighthearted_0"},
        {"rule": "Favor rounded display typefaces that smile rather than shout.", "id": "lighthearted_1"},
        {"rule": "Keep layouts airy and uncluttered to maintain a sense of ease.", "id": "lighthearted_2"},
    ],

    "spontaneous": [
        {"rule": "Break strict grid alignment occasionally to suggest energy and immediacy.", "id": "spontaneous_0"},
        {"rule": "Use saturated, unexpected color combinations that feel instinctive, not calculated.", "id": "spontaneous_1"},
        {"rule": "Introduce expressive typographic variation within a clear system.", "id": "spontaneous_2"},
    ],

    "luxury": [
        {"rule": "Use deep tones and metallic or neutral accent highlights.", "id": "luxury_0"},
        {"rule": "Prefer high-contrast type pairings with fine weight variation.", "id": "luxury_1"},
        {"rule": "Maximize whitespace; scarcity of elements signals scarcity of product.", "id": "luxury_2"},
    ],

    "elegant": [
        {"rule": "Maximize negative space and use type size contrast instead of decoration.", "id": "elegant_0"},
        {"rule": "Restrict the palette to two tones with one metallic or neutral accent.", "id": "elegant_1"},
        {"rule": "Use fine-weight serif or tall sans-serif faces for headline text.", "id": "elegant_2"},
    ],

    "sophisticated": [
        {"rule": "Use high-contrast type hierarchies with generous tracking on display text.", "id": "sophisticated_0"},
        {"rule": "Restrict accent colors to one; let form and proportion carry the weight.", "id": "sophisticated_1"},
        {"rule": "Use fine typographic details such as optical sizing or small caps where available.", "id": "sophisticated_2"},
    ],

    "timeless": [
        {"rule": "Avoid trend-driven typefaces; choose forms with proven readability.", "id": "timeless_0"},
        {"rule": "Use neutral, desaturated palettes that do not reference a specific decade.", "id": "timeless_1"},
        {"rule": "Rely on classical proportions and established compositional rules.", "id": "timeless_2"},
    ],

    "exclusive": [
        {"rule": "Use a very limited palette: black, white, and one precious-metal accent.", "id": "exclusive_0"},
        {"rule": "Maximize white space; scarcity of elements signals rarity.", "id": "exclusive_1"},
        {"rule": "Use fine-weight display type with wide tracking to signal exclusivity.", "id": "exclusive_2"},
    ],

    "heritage": [
        {"rule": "Reference historical typographic forms without pastiche.", "id": "heritage_0"},
        {"rule": "Use sepia, navy, burgundy, or aged-paper tones for a sense of time.", "id": "heritage_1"},
        {"rule": "Incorporate classical proportions and established serif traditions.", "id": "heritage_2"},
    ],

    "classic": [
        {"rule": "Use timeless typography and balanced traditional proportions.", "id": "classic_0"},
        {"rule": "Avoid trend-heavy decorative choices.", "id": "classic_1"},
        {"rule": "Reference established typographic conventions without literal pastiche.", "id": "classic_2"},
    ],

    "artisan": [
        {"rule": "Lean into texture cues and fonts that feel hand-crafted.", "id": "artisan_0"},
        {"rule": "Use muted, natural tones to signal authenticity and craft.", "id": "artisan_1"},
        {"rule": "Avoid overly polished or symmetrical layouts; slight irregularity signals humanity.", "id": "artisan_2"},
    ],

    "handmade": [
        {"rule": "Use typefaces with visible ink-trap details or hand-drawn qualities.", "id": "handmade_0"},
        {"rule": "Introduce deliberate texture and slight irregularity into layout and color application.", "id": "handmade_1"},
        {"rule": "Use warm, slightly impure color mixes rather than digitally pure hex values.", "id": "handmade_2"},
    ],

    "authentic": [
        {"rule": "Avoid overly polished or symmetrical layouts; slight irregularity signals humanity.", "id": "authentic_0"},
        {"rule": "Use typefaces with visible stroke variation or hand-crafted quality.", "id": "authentic_1"},
        {"rule": "Let natural imperfection show; perfect symmetry can undermine authenticity.", "id": "authentic_2"},
    ],

    "craft_driven": [
        {"rule": "Use typography that shows evidence of the designer's hand or optical correction.", "id": "craft_driven_0"},
        {"rule": "Draw the palette from natural craft materials such as wood, linen, clay, or ink.", "id": "craft_driven_1"},
        {"rule": "Allow asymmetry and visual weight imbalance to signal deliberate craft decision-making.", "id": "craft_driven_2"},
    ],

    "neighborly": [
        {"rule": "Use approachable, mid-weight sans or slab-serif typefaces that feel unpretentious.", "id": "neighborly_0"},
        {"rule": "Choose familiar warm palettes such as terracotta, cream, and soft green.", "id": "neighborly_1"},
        {"rule": "Keep layouts informal and conversational; avoid corporate rigidity.", "id": "neighborly_2"},
    ],

    "established": [
        {"rule": "Use classic serif typefaces with long publication histories.", "id": "established_0"},
        {"rule": "Anchor layouts in traditional proportions and proven grid systems.", "id": "established_1"},
        {"rule": "Use muted, historically grounded color palettes that avoid trend associations.", "id": "established_2"},
    ],

    "bold": [
        {"rule": "Use stronger contrast and a more assertive typographic scale.", "id": "bold_0"},
        {"rule": "Favor impactful focal points and simplified messaging.", "id": "bold_1"},
        {"rule": "Use high-saturation accent colors sparingly for maximum impact.", "id": "bold_2"},
    ],

    "authoritative": [
        {"rule": "Use a strong, high-contrast serif or condensed sans as the primary display face.", "id": "authoritative_0"},
        {"rule": "Anchor the palette with a deep, saturated primary color such as navy, forest, or burgundy.", "id": "authoritative_1"},
        {"rule": "Use large typographic scale contrasts to project confidence and command.", "id": "authoritative_2"},
    ],

    "intentional": [
        {"rule": "Every visual element must have an explicit reason for being present.", "id": "intentional_0"},
        {"rule": "Use restraint as the primary design tool; say more with less.", "id": "intentional_1"},
        {"rule": "Align typeface choices directly with the brand's core values rather than trend.", "id": "intentional_2"},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_attribute(value: str) -> str:
    """
    Normalize LLM-produced attributes into rule-bank-friendly keys.
    """
    attr = (value or "").strip().lower()
    attr = attr.replace("-", "_").replace(" ", "_")
    attr = attr.replace("&", "and")
    attr = attr.strip("_")
    if attr in ATTRIBUTE_ALIASES:
        return ATTRIBUTE_ALIASES[attr]

    for alias_key, alias_val in ATTRIBUTE_ALIASES.items():
        if attr.startswith(alias_key) or alias_key.startswith(attr):
            if len(attr) >= 4:   # 防止短词误匹配
                return alias_val

    # 直接命中 rule bank
    if attr in _RULE_BANK:
        return attr

    return attr

def _default_weights() -> Dict[str, float]:
    """
    Initialize every known rule weight to 1.0.
    """
    return {
        rule["id"]: 1.0
        for rules in _RULE_BANK.values()
        for rule in rules
    }


def _with_weights(
    rules: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Attach weights and sort descending.
    """
    w = weights or {}
    enriched = [
        {
            **rule,
            "weight": float(w.get(rule["id"], 1.0)),
        }
        for rule in rules
    ]
    return sorted(enriched, key=lambda r: r.get("weight", 1.0), reverse=True)


def _fallback_rules(attr: str) -> List[Dict[str, Any]]:
    """
    Better fallback rules when no direct rule-bank key exists.
    """
    return [
        {
            "rule": (
                f"Translate the attribute '{attr}' into concrete choices across "
                "layout, typography, color hierarchy, and spacing."
            ),
            "id": f"fallback_{attr}_0",
        },
        {
            "rule": (
                f"Keep visual decisions accountable to '{attr}' rather than adding "
                "decorative elements without purpose."
            ),
            "id": f"fallback_{attr}_1",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Core search function
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_search(
    brand_attribute: str,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve design rules for a brand attribute or archetype.

    Args:
        brand_attribute:
            Attribute produced by Agent 2, e.g. "premium", "modern",
            "eco-friendly", "precise", "craft-driven".
        weights:
            Optional learned rule weights from state["heuristic_weights"].
        top_k:
            Number of rules returned.

    Returns:
        [
            {"rule": "...", "id": "...", "weight": 1.0},
            ...
        ]
    """
    attr = _normalize_attribute(brand_attribute)

    if not attr:
        return []

    candidates = _RULE_BANK.get(attr)

    if candidates is None:
        candidates = _fallback_rules(attr)

    ranked = _with_weights(candidates, weights)
    return ranked[: max(1, int(top_k))]


def heuristics_to_generation_constraints(
    heuristics: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Convert retrieved heuristic rules into concrete generator constraints.

    This makes heuristic retrieval actionable instead of only explanatory.
    The returned constraints can be passed into color_retrieve, font_lookup,
    or later layout modules.
    """
    palette_constraints: List[str] = []
    font_constraints: List[str] = []
    layout_constraints: List[str] = []

    seen = set()

    def add(bucket: List[str], text: str) -> None:
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            bucket.append(text)

    for h in heuristics:
        rule_text = str(h.get("rule", "")).lower()
        tags = h.get("constraint_tags", []) or []

        # Prefer explicit tags when available.
        if "prefer_cool_neutral" in tags:
            add(
                palette_constraints,
                "Prefer cool neutral colors such as navy, slate, charcoal, blue, or grey.",
            )

        if "avoid_high_saturation" in tags:
            add(
                palette_constraints,
                "Avoid high-saturation or neon-like colors.",
            )

        if "one_controlled_accent" in tags:
            add(
                palette_constraints,
                "Use one controlled accent color against a neutral base.",
            )

        if "limited_palette" in tags:
            add(
                palette_constraints,
                "Use a limited palette with clearly assigned functional roles.",
            )

        if "bright_accent_allowed" in tags:
            add(
                palette_constraints,
                "Allow bright accent colors only if contrast remains readable.",
            )

        if "rounded_friendly_type" in tags:
            add(
                font_constraints,
                "Prefer rounded, friendly, highly readable typefaces.",
            )

        if "neutral_sans" in tags:
            add(
                font_constraints,
                "Prefer established neutral sans-serif typefaces.",
            )

        if "geometric_or_mono" in tags:
            add(
                font_constraints,
                "Prefer geometric sans or monospaced typefaces for precision.",
            )

        if "grid_layout" in tags:
            add(
                layout_constraints,
                "Use structured grid-based layouts with consistent spacing.",
            )

        if "generous_whitespace" in tags:
            add(
                layout_constraints,
                "Use generous whitespace and reduce visual clutter.",
            )

        # Backward-compatible fallback for rules without metadata.
        if not tags:
            if any(
                x in rule_text
                for x in [
                    "conservative blues",
                    "neutral anchors",
                    "deep navy",
                    "charcoal",
                    "cool blues",
                    "cool, desaturated palettes",
                ]
            ):
                add(
                    palette_constraints,
                    "Prefer cool neutral colors such as navy, slate, charcoal, blue, or grey.",
                )

            if any(
                x in rule_text
                for x in [
                    "avoid loud",
                    "avoid playful color",
                    "avoid warm or playful",
                    "high-saturation",
                    "highly saturated",
                    "neon",
                ]
            ):
                add(
                    palette_constraints,
                    "Avoid high-saturation or neon-like colors.",
                )

            if any(
                x in rule_text
                for x in [
                    "one controlled accent",
                    "single electric accent",
                    "one unexpected accent",
                    "single accent",
                ]
            ):
                add(
                    palette_constraints,
                    "Use one controlled accent color against a neutral base.",
                )

            if any(
                x in rule_text
                for x in [
                    "limited color palette",
                    "strictly limited color",
                    "limited palette",
                    "two or three carefully chosen colors",
                ]
            ):
                add(
                    palette_constraints,
                    "Use a limited palette with clearly assigned functional roles.",
                )

            if any(
                x in rule_text
                for x in [
                    "bright accents",
                    "brighter accents",
                    "vibrant accent",
                    "saturated accent",
                ]
            ):
                add(
                    palette_constraints,
                    "Allow bright accent colors only if contrast remains readable.",
                )

            if any(
                x in rule_text
                for x in [
                    "rounded",
                    "friendly typography",
                    "friendly display typefaces",
                ]
            ):
                add(
                    font_constraints,
                    "Prefer rounded, friendly, highly readable typefaces.",
                )

            if any(
                x in rule_text
                for x in [
                    "neutral families",
                    "strong legibility",
                    "established, neutral",
                    "functional sans-serif",
                ]
            ):
                add(
                    font_constraints,
                    "Prefer established neutral sans-serif typefaces.",
                )

            if any(
                x in rule_text
                for x in [
                    "monospaced",
                    "geometric typefaces",
                    "geometric sans",
                ]
            ):
                add(
                    font_constraints,
                    "Prefer geometric sans or monospaced typefaces for precision.",
                )

            if any(
                x in rule_text
                for x in [
                    "grid",
                    "alignment",
                    "consistent spacing",
                    "pixel-precise",
                ]
            ):
                add(
                    layout_constraints,
                    "Use structured grid-based layouts with consistent spacing.",
                )

            if any(
                x in rule_text
                for x in [
                    "whitespace",
                    "negative space",
                    "visual clutter",
                    "generous margins",
                ]
            ):
                add(
                    layout_constraints,
                    "Use generous whitespace and reduce visual clutter.",
                )

    return {
        "palette_constraints": palette_constraints,
        "font_constraints": font_constraints,
        "layout_constraints": layout_constraints,
    }


# ─────────────────────────────────────────────────────────────────────────────
# State helpers for LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def initialise_weights(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure state has heuristic_weights.
    Use this in planner_agent before Agent 2 runs.
    """
    existing = state.get("heuristic_weights")

    if isinstance(existing, dict) and existing:
        weights = _default_weights()
        weights.update(existing)
    else:
        weights = _default_weights()

    return {
        **state,
        "heuristic_weights": weights,
    }


def update_heuristic_weights(
    state: Dict[str, Any],
    qc_scores: Optional[Dict[str, Any]] = None,
    used_rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Update weights for heuristic rules based on QC result.

    The intuition:
    - If the draft performs better than BASELINE_SCORE, reward used rules.
    - If it performs worse, penalize used rules.
    - This is simple but enough to justify "self-improving heuristic search"
      in the project demo.

    Args:
        state:
            LangGraph shared state.
        qc_scores:
            Usually state["qc_scores"].
        used_rules:
            Usually kit["generator_trace"]["heuristics_used"].

    Returns:
        Updated state with heuristic_weights.
    """
    weights = dict(state.get("heuristic_weights") or _default_weights())
    qc_scores = qc_scores or state.get("qc_scores") or {}

    if used_rules is None:
        kit = state.get("draft_brand_kit") or state.get("approved_brand_kit") or {}
        trace = kit.get("generator_trace") or {}
        used_rules = trace.get("heuristics_used") or []

    try:
        overall_score = float(qc_scores.get("overall_score", BASELINE_SCORE * 5.0))
        if overall_score > 1.0:
            normalized_score = overall_score / 5.0
        else:
            normalized_score = overall_score
    except Exception:
        normalized_score = BASELINE_SCORE

    delta = LEARNING_RATE * (normalized_score - BASELINE_SCORE)

    for rule in used_rules:
        rule_id = rule.get("id")
        if not rule_id:
            continue

        old_weight = float(weights.get(rule_id, 1.0))
        new_weight = old_weight + delta
        new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
        weights[rule_id] = round(new_weight, 4)

    return {
        **state,
        "heuristic_weights": weights,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick manual test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_attrs = [
        "eco-friendly",
        "forward-thinking",
        "precise",
        "secure",
        "craftsmanship",
        "approachable",
        "unknown attribute",
    ]

    for attr in test_attrs:
        print(f"\nAttribute: {attr}")
        for r in heuristic_search(attr):
            print(f"  - [{r['id']}] {r['rule']} | weight={r['weight']}")

    print("\nConstraint conversion smoke test:")
    sample_rules = heuristic_search("corporate") + heuristic_search("precise")
    converted = heuristics_to_generation_constraints(sample_rules)
    print(converted)
