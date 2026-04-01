def heuristic_search(brand_attribute: str):
    attr = (brand_attribute or "").strip().lower()

    rule_bank = {
        "premium": [
            {"rule": "Use restrained color contrast and generous whitespace."},
            {"rule": "Prefer elegant typography with a refined visual hierarchy."},
        ],
        "refined": [
            {"rule": "Reduce visual clutter and keep layouts highly ordered."},
            {"rule": "Use subtle accents instead of loud, high-saturation highlights."},
        ],
        "modern": [
            {"rule": "Favor clean alignment, simple geometry, and minimal ornament."},
            {"rule": "Use contemporary sans or high-contrast serif pairings."},
        ],
        "trustworthy": [
            {"rule": "Maintain strong readability and consistent spacing throughout."},
            {"rule": "Favor balanced compositions and stable, calm visual rhythm."},
        ],
        "soft": [
            {"rule": "Use gentle tonal transitions and avoid harsh contrast jumps."},
            {"rule": "Favor rounded or graceful forms and a calm palette."},
        ],
        "playful": [
            {"rule": "Allow brighter accents and more energetic composition."},
            {"rule": "Use friendlier typography and slightly more visual motion."},
        ],
        "organic": [
            {"rule": "Favor earthy or natural hues and softer saturation."},
            {"rule": "Use human, tactile typography and less rigid composition."},
        ],
        "bold": [
            {"rule": "Use stronger contrast and a more assertive typographic scale."},
            {"rule": "Favor impactful focal points and simplified messaging."},
        ],
        "classic": [
            {"rule": "Use timeless typography and balanced traditional proportions."},
            {"rule": "Avoid trend-heavy decorative choices."},
        ],
    }

    default_rules = [
        {"rule": f"Use layout, typography, and color choices to reinforce {attr}."},
        {"rule": f"Keep the overall visual system consistent with the attribute {attr}."},
    ]

    return rule_bank.get(attr, default_rules)
