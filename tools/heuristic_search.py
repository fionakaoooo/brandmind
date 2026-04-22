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


# ── Default weight initialiser ────────────────────────────────────────────────

def _default_weights() -> Dict[str, float]:
    """
    初始化所有规则权重为 1.0
    """
    return {
        rule["id"]: 1.0
        for rules in _RULE_BANK.values()
        for rule in rules
    }


# ── Core search function ──────────────────────────────────────────────────────

def heuristic_search(
    brand_attribute: str,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    按当前权重返回某个 brand_attribute 对应的设计规则。

    返回格式：
    [
        {
            "rule": "...",
            "id": "...",
            "weight": 1.0
        },
        ...
    ]
    """
    attr = (brand_attribute or "").strip().lower()
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


# ── Weight initialisation helper ──────────────────────────────────────────────

def initialise_weights(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    确保 state 中存在 heuristic_weights。
    只在缺失时初始化，不覆盖已有权重。
    """
    if not state.get("heuristic_weights"):
        state = {**state, "heuristic_weights": _default_weights()}
    return state


# ── Helper: find which rules were actually used ───────────────────────────────

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


# ── Helper: compute continuous feedback from qc_scores ────────────────────────

def _compute_feedback_score(qc_scores: Dict[str, Any]) -> float:
    """
    从 Agent 3 的 qc_scores 里构造连续反馈分数，范围大致在 [0, 1]。

    结构兼容你现在的 qc_agent:
    {
      "wcag": {"pass_rate": ...},
      "coherence": {"score": ...},     # 1~5
      "constraints": {"pass_rate": ...},
      ...
    }
    """
    if not qc_scores:
        return BASELINE_SCORE

    wcag_pass_rate = float(qc_scores.get("wcag", {}).get("pass_rate", 0.0))
    coherence_score = float(qc_scores.get("coherence", {}).get("score", 0.0)) / 5.0
    constraint_pass_rate = float(qc_scores.get("constraints", {}).get("pass_rate", 0.0))

    # 连续反馈：不是 pass/fail 二元，而是综合三项质量
    feedback_score = (
        0.40 * wcag_pass_rate
        + 0.35 * coherence_score
        + 0.25 * constraint_pass_rate
    )

    # 防守性裁剪
    feedback_score = max(0.0, min(1.0, feedback_score))
    return round(feedback_score, 4)


# ── Main update function ──────────────────────────────────────────────────────

def update_heuristic_weights(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    用 qc_scores 连续更新当前 draft 用到的 heuristic rule 权重。

    更新逻辑：
        delta = LEARNING_RATE * (feedback_score - BASELINE_SCORE)

    含义：
    - 如果 feedback_score > baseline，权重上升
    - 如果 feedback_score < baseline，权重下降
    """
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

    return {
        **state,
        "heuristic_weights": weights,
    }


# ── Utility: inspect top rules ────────────────────────────────────────────────

def get_top_rules(weights: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    返回当前全局权重最高的 top_k 条规则，便于调试和可视化。
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


# ── Utility: inspect a single attribute's rules ───────────────────────────────

def inspect_attribute_rules(
    attribute: str,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    查看某个 attribute 下规则当前权重排序结果。
    """
    return heuristic_search(attribute, weights=weights)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_state: Dict[str, Any] = {
        "heuristic_weights": None,
        "draft_brand_kit": {
            "design_rules": [
                "Use restrained color contrast and generous whitespace.",   # premium_0
                "Favor earthy or natural hues and softer saturation.",      # organic_0
            ]
        },
        "qc_scores": {
            "wcag": {"pass_rate": 0.60},
            "coherence": {"score": 3.5},
            "constraints": {"pass_rate": 0.67},
        },
    }

    mock_state = initialise_weights(mock_state)
    print("Initial weights (sample):", list(mock_state["heuristic_weights"].items())[:5])

    mock_state = update_heuristic_weights(mock_state)

    print("\nUpdated premium rules:")
    for r in inspect_attribute_rules("premium", weights=mock_state["heuristic_weights"]):
        print(f"  [{r['weight']:.4f}] {r['id']} -> {r['rule']}")

    print("\nUpdated organic rules:")
    for r in inspect_attribute_rules("organic", weights=mock_state["heuristic_weights"]):
        print(f"  [{r['weight']:.4f}] {r['id']} -> {r['rule']}")

    print("\nTop 5 rules overall:")
    for entry in get_top_rules(mock_state["heuristic_weights"], top_k=5):
        print(f"  [{entry['weight']:.4f}] ({entry['attribute']}) {entry['rule']}")
