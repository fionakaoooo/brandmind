
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Reward / penalty hyperparameters ──────────────────────────────────────────

REWARD = 0.15    # weight increase when a rule contributes to an approved kit
PENALTY = 0.08   # weight decrease when a rule contributes to a failed kit
MIN_WEIGHT = 0.10
MAX_WEIGHT = 3.00

# ── Static rule bank ───────────────────────────────────────────────────────────
# Every rule has a stable "id" so weights can be tracked across sessions.

_RULE_BANK: Dict[str, List[Dict[str, Any]]] = {
    "premium": [
        {"rule": "Use restrained color contrast and generous whitespace.", "id": "premium_0"},
        {"rule": "Prefer elegant typography with a refined visual hierarchy.", "id": "premium_1"},
    ],
    "refined": [
        {"rule": "Reduce visual clutter and keep layouts highly ordered.", "id": "refined_0"},
        {"rule": "Use subtle accents instead of loud, high-saturation highlights.", "id": "refined_1"},
    ],
    "modern": [
        {"rule": "Favor clean alignment, simple geometry, and minimal ornament.", "id": "modern_0"},
        {"rule": "Use contemporary sans or high-contrast serif pairings.", "id": "modern_1"},
    ],
    "trustworthy": [
        {"rule": "Maintain strong readability and consistent spacing throughout.", "id": "trustworthy_0"},
        {"rule": "Favor balanced compositions and stable, calm visual rhythm.", "id": "trustworthy_1"},
    ],
    "soft": [
        {"rule": "Use gentle tonal transitions and avoid harsh contrast jumps.", "id": "soft_0"},
        {"rule": "Favor rounded or graceful forms and a calm palette.", "id": "soft_1"},
    ],
    "playful": [
        {"rule": "Allow brighter accents and more energetic composition.", "id": "playful_0"},
        {"rule": "Use friendlier typography and slightly more visual motion.", "id": "playful_1"},
    ],
    "organic": [
        {"rule": "Favor earthy or natural hues and softer saturation.", "id": "organic_0"},
        {"rule": "Use human, tactile typography and less rigid composition.", "id": "organic_1"},
    ],
    "bold": [
        {"rule": "Use stronger contrast and a more assertive typographic scale.", "id": "bold_0"},
        {"rule": "Favor impactful focal points and simplified messaging.", "id": "bold_1"},
    ],
    "classic": [
        {"rule": "Use timeless typography and balanced traditional proportions.", "id": "classic_0"},
        {"rule": "Avoid trend-heavy decorative choices.", "id": "classic_1"},
    ],
    "minimal": [
        {"rule": "Maximise negative space and resist adding decorative detail.", "id": "minimal_0"},
        {"rule": "Limit the active palette to two or three carefully chosen colours.", "id": "minimal_1"},
    ],
    "luxury": [
        {"rule": "Use deep tones and metallic or neutral accent highlights.", "id": "luxury_0"},
        {"rule": "Prefer high-contrast type pairings with fine weight variation.", "id": "luxury_1"},
    ],
    "tech": [
        {"rule": "Use cool, high-contrast colours with strong geometric forms.", "id": "tech_0"},
        {"rule": "Prefer monospaced or geometric sans fonts for data contexts.", "id": "tech_1"},
    ],
    "artisan": [
        {"rule": "Lean into texture cues: use fonts that feel hand-crafted.", "id": "artisan_0"},
        {"rule": "Muted, natural tones signal authenticity and craft.", "id": "artisan_1"},
    ],
    "heritage": [
        {"rule": "Reference historical typographic forms without pastiche.", "id": "heritage_0"},
        {"rule": "Use sepia, navy, or aged-paper tones for a sense of time.", "id": "heritage_1"},
    ],
    "youthful": [
        {"rule": "Use vibrant accent colours offset by clean, open space.", "id": "youthful_0"},
        {"rule": "Favour rounded, friendly display typefaces over rigid geometry.", "id": "youthful_1"},
    ],
    "corporate": [
        {"rule": "Prioritise legibility and ordered grid-based layouts.", "id": "corporate_0"},
        {"rule": "Use conservative blues or greys to project stability.", "id": "corporate_1"},
    ],
}


# ── Default weight initialiser ─────────────────────────────────────────────────

def _default_weights() -> Dict[str, float]:
    """Return a weight dict keyed by rule id, all initialised to 1.0."""
    return {
        rule["id"]: 1.0
        for rules in _RULE_BANK.values()
        for rule in rules
    }


# ── Core search function ───────────────────────────────────────────────────────

def heuristic_search(
    brand_attribute: str,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Return design rules for `brand_attribute`, sorted by current weight descending.

    Parameters
    ----------
    brand_attribute : str
        A single brand attribute, e.g. "premium", "organic", "trustworthy".
    weights : dict, optional
        The `heuristic_weights` dict from BrandMindState.
        Pass None (or omit) to treat all weights as 1.0.

    Returns
    -------
    List of dicts, each with:
        "rule"   – the design rule text
        "id"     – stable identifier used for weight tracking
        "weight" – current weight (included for logging/debugging)
    """
    attr = (brand_attribute or "").strip().lower()
    w = weights or {}

    candidates = _RULE_BANK.get(attr)

    if candidates is None:
        # Fallback: generic rules for unknown attributes
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
        return sorted(
            [{**r, "weight": w.get(r["id"], 1.0)} for r in fallback],
            key=lambda r: r["weight"],
            reverse=True,
        )

    enriched = [{**rule, "weight": w.get(rule["id"], 1.0)} for rule in candidates]
    return sorted(enriched, key=lambda r: r["weight"], reverse=True)


# ── Weight initialisation helper ───────────────────────────────────────────────

def initialise_weights(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures `heuristic_weights` exists in state. Safe to call even if weights
    are already present — existing data is never overwritten.

    Call this at the very start of the pipeline (e.g. top of planner_agent).
    """
    if not state.get("heuristic_weights"):
        state = {**state, "heuristic_weights": _default_weights()}
    return state


# ── Weight update ──────────────────────────────────────────────────────────────

def update_heuristic_weights(
    state: Dict[str, Any],
    approved: bool,
) -> Dict[str, Any]:
    """
    Adjust rule weights based on QC outcome for the current draft.

    Call this at the end of qc_agent() before returning the updated state,
    passing the `approved` boolean from the QC decision.

    Parameters
    ----------
    state    : current BrandMindState
    approved : True  → reward rules used in this draft
               False → penalise rules used in this draft

    Returns
    -------
    Updated state dict with modified `heuristic_weights`.
    """
    weights: Dict[str, float] = dict(state.get("heuristic_weights") or _default_weights())
    used_ids = _extract_used_rule_ids(state)

    if not used_ids:
        return state  # nothing to update if no rules were recorded

    delta = REWARD if approved else -PENALTY

    for rule_id in used_ids:
        old = weights.get(rule_id, 1.0)
        weights[rule_id] = round(max(MIN_WEIGHT, min(MAX_WEIGHT, old + delta)), 4)

    print(
        f"[HeuristicSearch] Weight {'reward' if approved else 'penalty'} "
        f"(Δ={delta:+.2f}) applied to {len(used_ids)} rule(s)."
    )

    return {**state, "heuristic_weights": weights}


# ── Helper: find rule ids used in the current draft ────────────────────────────

def _extract_used_rule_ids(state: Dict[str, Any]) -> List[str]:
    """
    Reads design_rules from the draft_brand_kit and reverse-maps rule text → id
    so we know exactly which rules to reward or penalise.
    """
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


# ── Utility: inspect top rules (handy for Streamlit / debugging) ───────────────

def get_top_rules(weights: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Return the top_k rules across the entire bank, sorted by weight descending.
    """
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


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate state with no prior weights
    state: Dict[str, Any] = {
        "heuristic_weights": None,
        "draft_brand_kit": {
            "design_rules": [
                "Use restrained color contrast and generous whitespace.",   # premium_0
                "Favor earthy or natural hues and softer saturation.",      # organic_0
            ]
        },
    }

    state = initialise_weights(state)
    print("Defaults (first 5):", list(state["heuristic_weights"].items())[:5])

    # Failed QC → penalise
    state = update_heuristic_weights(state, approved=False)
    print(f"\nAfter penalty  — premium_0: {state['heuristic_weights']['premium_0']:.4f}  "
          f"organic_0: {state['heuristic_weights']['organic_0']:.4f}")

    # Two successful passes → reward twice
    state = update_heuristic_weights(state, approved=True)
    state = update_heuristic_weights(state, approved=True)
    print(f"After 2x reward — premium_0: {state['heuristic_weights']['premium_0']:.4f}  "
          f"organic_0: {state['heuristic_weights']['organic_0']:.4f}")

    # Search with updated weights
    print("\nOrganic rules (ranked by weight):")
    for r in heuristic_search("organic", weights=state["heuristic_weights"]):
        print(f"  [{r['weight']:.3f}] {r['rule']}")

    print("\nTop 5 rules across all attributes:")
    for entry in get_top_rules(state["heuristic_weights"], top_k=5):
        print(f"  [{entry['weight']:.3f}] ({entry['attribute']}) {entry['rule']}")
