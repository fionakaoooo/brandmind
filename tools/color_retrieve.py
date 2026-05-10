from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import pandas as pd


PALETTE_CSV_PATH = os.environ.get(
    "BRAND_PALETTE_CSV",
    "data/processed/branding_palettes_processed.csv"
)

EMOSET_SUMMARY_PATH = os.environ.get(
    "EMOSET_SUMMARY_PATH",
    "data/processed/emoset_emotion_summary.csv"
)

HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{6})$")


EMOSET_BRAND_EMOTION_MAP = {
    "calm": ["contentment"],
    "soft": ["contentment"],
    "elegant": ["awe", "contentment"],
    "sophisticated": ["awe"],
    "modern": ["awe"],
    "playful": ["amusement", "excitement"],
    "energetic": ["excitement"],
    "bold": ["excitement", "anger"],
    "warm": ["contentment", "amusement"],
    "trustworthy": ["contentment"],
    "refined": ["awe", "contentment"],
    "luxurious": ["awe"],
    "premium": ["awe"],
    "friendly": ["amusement", "contentment"],
    "joyful": ["amusement", "excitement"],
    "grounded": ["contentment"],
    "precise":       ["awe"],
    "reliable":      ["contentment"],
    "innovative":    ["awe", "excitement"],
    "efficient":     ["contentment"],
    "secure":        ["contentment"],
    "professional":  ["contentment"],
    "authoritative": ["awe"],
    "contemporary":  ["awe"],
    "soothing":      ["contentment"],
    "nurturing":     ["contentment", "amusement"],
    "natural":       ["contentment"],
    "timeless":      ["awe", "contentment"],
    "exclusive":     ["awe"],
    "curious":       ["amusement", "excitement"],
    "imaginative":   ["amusement", "excitement"],
    "approachable":  ["amusement", "contentment"],
    "dynamic":       ["excitement"],
    "authentic":     ["contentment"],
    "craft_driven":  ["contentment", "awe"],
    "nostalgic":     ["contentment", "awe"],
    "handmade":      ["contentment"],
    "rebellious":    ["excitement", "anger"],
    "technical":     ["awe"],
    "cutting_edge":  ["awe", "excitement"],
}

def _normalize_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


@lru_cache(maxsize=1)
def load_palette_df() -> pd.DataFrame:
    if not os.path.exists(PALETTE_CSV_PATH):
        raise FileNotFoundError(
            f"Palette CSV not found at {PALETTE_CSV_PATH}. "
            "Set BRAND_PALETTE_CSV or create the processed file first."
        )
    df = pd.read_csv(PALETTE_CSV_PATH)
    df.columns = [_normalize_col(c) for c in df.columns]
    return df


@lru_cache(maxsize=1)
def load_emoset_summary() -> pd.DataFrame:
    if not os.path.exists(EMOSET_SUMMARY_PATH):
        return pd.DataFrame(columns=["emotion", "brightness_mean", "colorfulness_mean"])
    df = pd.read_csv(EMOSET_SUMMARY_PATH)
    df.columns = [_normalize_col(c) for c in df.columns]
    return df


def _find_hex_columns(df: pd.DataFrame) -> List[str]:
    hex_cols = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20).tolist()
        if sample and sum(bool(HEX_RE.match(x.strip())) for x in sample) >= max(3, len(sample) // 2):
            hex_cols.append(col)
    if len(hex_cols) >= 5:
        return hex_cols[:5]
    common = [c for c in ["color1", "color2", "color3", "color4", "color5"] if c in df.columns]
    return common


def _find_label_columns(df: pd.DataFrame, hex_cols: List[str]) -> List[str]:
    excluded = set(hex_cols) | {"id", "palette_id", "palette_name", "brand", "industry"}
    label_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        vals = set(df[col].dropna().astype(str).str.lower().unique().tolist())
        if vals.issubset({"0", "1", "0.0", "1.0", "true", "false"}):
            label_cols.append(col)
    return label_cols


def _to_binary(val: Any) -> int:
    sval = str(val).strip().lower()
    return 1 if sval in {"1", "1.0", "true"} else 0


def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = str(hex_code).strip().lstrip("#")
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(hex_code: str) -> float:
    rgb = _hex_to_rgb(hex_code)

    def channel(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [channel(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = _relative_luminance(hex1)
    l2 = _relative_luminance(hex2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _pairwise_wcag_stats(hex_codes: List[str]) -> Dict[str, Any]:
    valid = [h.strip() for h in hex_codes if HEX_RE.match(str(h).strip())]
    if len(valid) < 2:
        return {
            "pass_count": 0,
            "total_pairs": 0,
            "pass_rate": 0.0,
            "max_ratio": 0.0,
            "min_ratio": 0.0,
        }

    pass_count = 0
    total_pairs = 0
    ratios: List[float] = []

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ratio = _contrast_ratio(valid[i], valid[j])
            ratios.append(ratio)
            total_pairs += 1
            if ratio >= 4.5:
                pass_count += 1

    return {
        "pass_count": pass_count,
        "total_pairs": total_pairs,
        "pass_rate": round(pass_count / total_pairs, 3) if total_pairs else 0.0,
        "max_ratio": round(max(ratios), 3) if ratios else 0.0,
        "min_ratio": round(min(ratios), 3) if ratios else 0.0,
    }


def _rgb_to_hsv_scaled(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
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


def _palette_stats(hex_codes: List[str]) -> Dict[str, float]:
    hsvs = [_rgb_to_hsv_scaled(_hex_to_rgb(h)) for h in hex_codes if HEX_RE.match(str(h))]
    if not hsvs:
        return {
            "avg_hue": 0.0,
            "avg_saturation": 0.0,
            "avg_brightness": 0.0,
            "avg_colorfulness": 0.0,
        }

    avg_h = sum(h for h, _, _ in hsvs) / len(hsvs)
    avg_s = sum(s for _, s, _ in hsvs) / len(hsvs)
    avg_v = sum(v for _, _, v in hsvs) / len(hsvs)

    return {
        "avg_hue": avg_h,
        "avg_saturation": avg_s,
        "avg_brightness": avg_v,
        "avg_colorfulness": avg_s,
    }


def _build_emoset_profile(emotions: List[str]) -> Dict[str, float]:
    df = load_emoset_summary()
    if df.empty:
        return {"brightness_target": 0.55, "colorfulness_target": 0.55}

    emotions_norm = [_normalize_col(e) for e in emotions]

    direct_hit = df[df["emotion"].astype(str).map(_normalize_col).isin(set(emotions_norm))]
    if not direct_hit.empty:
        return {
            "brightness_target": float(direct_hit["brightness_mean"].mean()),
            "colorfulness_target": float(direct_hit["colorfulness_mean"].mean()),
        }

    mapped: List[str] = []
    for emo in emotions_norm:
        mapped.extend(EMOSET_BRAND_EMOTION_MAP.get(emo, []))
    mapped = [_normalize_col(x) for x in mapped]

    mapped_hit = df[df["emotion"].astype(str).map(_normalize_col).isin(set(mapped))]
    if not mapped_hit.empty:
        return {
            "brightness_target": float(mapped_hit["brightness_mean"].mean()),
            "colorfulness_target": float(mapped_hit["colorfulness_mean"].mean()),
        }

    return {"brightness_target": 0.55, "colorfulness_target": 0.55}


def _distance(a: float, b: float) -> float:
    return abs(a - b)


def _industry_bonus(row: pd.Series, industry: str) -> float:
    if "industry" not in row.index:
        return 0.0
    row_industry = str(row.get("industry", "")).strip().lower()
    query_industry = industry.strip().lower()
    if not row_industry or not query_industry:
        return 0.0
    return 1.5 if query_industry in row_industry else 0.0

def _constraint_penalty(
    hex_codes: List[str],
    stats: Dict[str, float],
    constraints: List[str],
) -> float:
    """
    Penalize palettes that violate explicit user constraints or
    heuristic-derived generation constraints.

    This version recognizes constraints produced by
    heuristics_to_generation_constraints(), such as:
    - Prefer cool neutral colors...
    - Avoid high-saturation or neon-like colors.
    - Use one controlled accent color against a neutral base.
    - Use a limited palette...
    """
    penalty = 0.0
    constraint_text = " ".join(str(c).lower() for c in constraints)

    prefer_cool_neutral = (
        "prefer cool neutral" in constraint_text
        or "navy" in constraint_text
        or "slate" in constraint_text
        or "charcoal" in constraint_text
        or "blue, or grey" in constraint_text
        or "blue or grey" in constraint_text
        or "cool blues" in constraint_text
        or "cool neutral colors" in constraint_text
    )

    avoid_high_saturation = (
        "avoid high-saturation" in constraint_text
        or "avoid high saturation" in constraint_text
        or "neon-like" in constraint_text
        or "avoid neon" in constraint_text
        or "no neon" in constraint_text
        or "avoid harsh colors" in constraint_text
        or "overly bright" in constraint_text
        or "harsh colors" in constraint_text
    )

    one_controlled_accent = (
        "one controlled accent" in constraint_text
        or "neutral base" in constraint_text
        or "single accent" in constraint_text
        or "controlled accent" in constraint_text
    )

    limited_palette = (
        "limited palette" in constraint_text
        or "limited color palette" in constraint_text
        or "clearly assigned functional roles" in constraint_text
        or "minimal" in constraint_text
    )
    playful_or_kids = (
        "playful" in constraint_text
        or "children" in constraint_text
        or "kids" in constraint_text
        or "toy" in constraint_text
        or "bright colors" in constraint_text
    )

    corporate_or_professional = (
        "corporate" in constraint_text
        or "professional" in constraint_text
        or "enterprise" in constraint_text
        or "fintech" in constraint_text
        or "b2b" in constraint_text
        or "trustworthy" in constraint_text
        or "precise" in constraint_text
    )

    warm_or_earthy_requested = (
        "warm" in constraint_text
        or "earthy" in constraint_text
        or "organic" in constraint_text
        or "natural" in constraint_text
        or "artisan" in constraint_text
    )

    accent_like_count = 0
    very_saturated_count = 0
    cool_neutral_like_count = 0

    for h in hex_codes:
        if not HEX_RE.match(str(h).strip()):
            continue

        hue, sat, val = _rgb_to_hsv_scaled(_hex_to_rgb(h))

        is_red = hue < 25 or hue > 335
        is_orange_yellow = 25 <= hue <= 70
        is_green_yellow = 70 < hue <= 170
        is_cyan_blue = 170 < hue <= 260
        is_purple_magenta = 260 < hue <= 335
        is_neutralish = sat <= 0.18
        is_dark_anchor = val <= 0.32
        is_light_neutral = sat <= 0.20 and val >= 0.86
        is_cool_hue = 185 <= hue <= 245

        if is_neutralish or is_dark_anchor or is_light_neutral or (is_cool_hue and sat <= 0.65):
            cool_neutral_like_count += 1

        if sat > 0.65:
            very_saturated_count += 1

        if sat > 0.55 and val > 0.55:
            accent_like_count += 1

        # Explicit red restriction.
        if "no red" in constraint_text or "avoid red" in constraint_text:
            if is_red:
                penalty += 1.5

        # General harsh/neon restriction.
        if avoid_high_saturation:
            if sat > 0.65:
                penalty += 1.2
            if sat > 0.72 and val > 0.80:
                penalty += 1.8
            if val > 0.88 and sat > 0.45:
                penalty += 0.8

        # Corporate / professional / fintech should avoid loud playful drift.
        if corporate_or_professional:
            if sat > 0.70 and val > 0.65:
                penalty += 1.4
            if is_purple_magenta and sat > 0.45:
                penalty += 0.9
            if is_green_yellow and sat > 0.45:
                penalty += 0.9
            if is_red and sat > 0.35:
                penalty += 0.8

        # Heuristic-derived cool neutral preference.
        if prefer_cool_neutral:
            if sat > 0.70 and val > 0.60:
                penalty += 1.3

            # Purple/magenta and yellow-green often hurt corporate trust tone.
            if is_purple_magenta and sat > 0.40:
                penalty += 1.0
            if is_green_yellow and sat > 0.45:
                penalty += 1.0
            if is_red and sat > 0.35:
                penalty += 0.8

            # Warm colors are only penalized if the brief did not request warmth/earthiness.
            if not warm_or_earthy_requested and is_orange_yellow and sat > 0.45:
                penalty += 0.7

        # Soft/calm/premium constraints still discourage saturation.
        if "soft" in constraint_text or "calm" in constraint_text:
            if sat > 0.60:
                penalty += 0.8

        if "luxury" in constraint_text or "premium" in constraint_text:
            if sat > 0.65:
                penalty += 0.6

    # Palette-level penalties.
    if avoid_high_saturation:
        if stats["avg_colorfulness"] > 0.55:
            penalty += 2.0
        if stats["avg_brightness"] > 0.82 and stats["avg_colorfulness"] > 0.45:
            penalty += 1.0

    if prefer_cool_neutral:
        # Require at least two colors that can plausibly serve as neutral/cool anchors.
        if cool_neutral_like_count < 2:
            penalty += 2.0

        # Corporate palettes should not be mostly saturated accents.
        if very_saturated_count >= 3:
            penalty += 2.5

    if one_controlled_accent:
        # More than two accent-like colors violates "one controlled accent".
        if accent_like_count > 2:
            penalty += (accent_like_count - 2) * 1.4

    if limited_palette:
        # Penalize palettes where too many colors are visually loud.
        if very_saturated_count >= 3:
            penalty += 1.8

    if "soft" in constraint_text or "calm" in constraint_text:
        if stats["avg_colorfulness"] > 0.50:
            penalty += 1.2

    # Playful/kids can tolerate saturation, so reduce over-penalization slightly.
    if playful_or_kids and not corporate_or_professional:
        penalty *= 0.75

    return penalty


def color_retrieve(
    emotions: List[str],
    industry: str = "",
    style_keywords: List[str] | None = None,
    constraints: List[str] | None = None,
    top_k: int = 5,
    excluded_hex: List[str] | None = None,
) -> Dict[str, Any]:
    style_keywords = style_keywords or []
    constraints = constraints or []

    excluded_set: set[str] = set()
    if excluded_hex:
        for h in excluded_hex:
            bare = str(h).strip().lstrip("#").upper()
            if len(bare) == 6:
                excluded_set.add(bare)

    df = load_palette_df().copy()
    hex_cols = _find_hex_columns(df)
    label_cols = _find_label_columns(df, hex_cols)

    if len(hex_cols) < 5:
        raise ValueError(
            "Could not detect five hex color columns in the palette dataset. "
            "Please preprocess the CSV into a stable format."
        )

    requested_terms = {_normalize_col(x) for x in (emotions + style_keywords)}
    emoset_profile = _build_emoset_profile(emotions)
    constraint_text = " ".join(str(c).lower() for c in constraints)

    requires_wcag = any(
        kw in constraint_text
        for kw in ["wcag", "accessible", "accessibility", "contrast", "colorblind"]
    )

    prefer_cool_neutral = (
        "prefer cool neutral" in constraint_text
        or "navy" in constraint_text
        or "slate" in constraint_text
        or "charcoal" in constraint_text
        or "cool neutral colors" in constraint_text
    )

    avoid_high_saturation = (
        "avoid high-saturation" in constraint_text
        or "avoid high saturation" in constraint_text
        or "avoid neon" in constraint_text
        or "no neon" in constraint_text
        or "neon-like" in constraint_text
        or "avoid harsh colors" in constraint_text
        or "overly bright" in constraint_text
        or "harsh colors" in constraint_text
    )

    one_controlled_accent = (
        "one controlled accent" in constraint_text
        or "neutral base" in constraint_text
        or "single accent" in constraint_text
    )

    scored_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_labels = [c for c in label_cols if _to_binary(row[c]) == 1]
        row_label_set = set(row_labels)

        emotion_hits = len(requested_terms & row_label_set)
        base_score = float(emotion_hits) * 3.0
        industry_score = _industry_bonus(row, industry)

        hex_codes = [str(row[c]).strip() for c in hex_cols]
        valid_hex = [h for h in hex_codes if HEX_RE.match(str(h).strip())]
        stats = _palette_stats(valid_hex)

        brightness_gap = _distance(
            stats["avg_brightness"],
            emoset_profile["brightness_target"],
        )
        colorfulness_gap = _distance(
            stats["avg_colorfulness"],
            emoset_profile["colorfulness_target"],
        )
        emoset_alignment_score = max(
            0.0,
            2.0 - (brightness_gap + colorfulness_gap) * 2.5,
        )

        penalty = _constraint_penalty(
            hex_codes=valid_hex,
            stats=stats,
            constraints=constraints,
        )

        wcag_stats = _pairwise_wcag_stats(valid_hex)
        wcag_score = wcag_stats["pass_rate"] * 4.0

        # Extra soft bonuses for heuristic-derived constraints.
        cool_neutral_bonus = 0.0
        controlled_accent_bonus = 0.0

        if valid_hex:
            hsvs = [_rgb_to_hsv_scaled(_hex_to_rgb(h)) for h in valid_hex]

            neutral_or_cool_count = 0
            accent_like_count = 0

            for hue, sat, val in hsvs:
                is_neutralish = sat <= 0.18
                is_dark_anchor = val <= 0.32
                is_light_neutral = sat <= 0.20 and val >= 0.86
                is_cool_hue = 185 <= hue <= 245 and sat <= 0.65

                if is_neutralish or is_dark_anchor or is_light_neutral or is_cool_hue:
                    neutral_or_cool_count += 1

                if sat > 0.55 and val > 0.55:
                    accent_like_count += 1

            if prefer_cool_neutral:
                cool_neutral_bonus = min(1.5, neutral_or_cool_count * 0.35)

            if one_controlled_accent and accent_like_count <= 2:
                controlled_accent_bonus = 0.8

        total_score = (
            base_score
            + industry_score
            + emoset_alignment_score
            + wcag_score
            + cool_neutral_bonus
            + controlled_accent_bonus
            - (penalty * 2.0)
        )

        # Stronger WCAG enforcement than before.
        # Your previous version only deducted 2 points for pass_rate < 0.50,
        # which allowed weak palettes to keep winning. 
        if requires_wcag:
            if wcag_stats["pass_rate"] < 0.50:
                total_score -= 8.0
            elif wcag_stats["pass_rate"] < 0.70:
                total_score -= 3.5

        # Stronger heuristic enforcement.
        if prefer_cool_neutral and penalty >= 3.0:
            total_score -= 4.0

        if avoid_high_saturation and penalty >= 3.0:
            total_score -= 4.0

        # If a corporate/cool-neutral palette has too many loud accents, push it down.
        if one_controlled_accent and penalty >= 3.0:
            total_score -= 2.0

        scored_rows.append({
            "palette_name": row.get(
                "palette_name",
                row.get("name", f"palette_{len(scored_rows) + 1}"),
            ),
            "hex_codes": hex_codes,
            "matched_emotions": sorted(list(requested_terms & row_label_set)),
            "available_labels": sorted(row_labels),
            "emotion_score": round(base_score, 3),
            "industry_bonus": round(industry_score, 3),
            "emoset_alignment": {
                "brightness_target": round(emoset_profile["brightness_target"], 3),
                "colorfulness_target": round(emoset_profile["colorfulness_target"], 3),
                "palette_brightness": round(stats["avg_brightness"], 3),
                "palette_colorfulness": round(stats["avg_colorfulness"], 3),
                "alignment_score": round(emoset_alignment_score, 3),
            },
            "wcag_stats": wcag_stats,
            "penalty": round(penalty, 3),
            "heuristic_bonus": {
                "cool_neutral_bonus": round(cool_neutral_bonus, 3),
                "controlled_accent_bonus": round(controlled_accent_bonus, 3),
            },
            "total_score": round(total_score, 3),
        })

    scored_rows.sort(key=lambda x: x["total_score"], reverse=True)

    if excluded_set:
        for row in scored_rows:
            normalized = {h.upper().strip().lstrip("#") for h in row["hex_codes"]}
            overlap = sum(1 for h in normalized if h in excluded_set)
            if overlap >= 1:
                row["total_score"] -= overlap * 5.0

        scored_rows.sort(key=lambda x: x["total_score"], reverse=True)
        print(
            f"[ColorRetrieve] Applied exclusion penalty for "
            f"{len(excluded_set)} colors from previous palette."
        )

    top = scored_rows[:top_k]
    best = top[0] if top else {}

    rationale = (
        f"Retrieved palettes by matching requested emotions/styles {sorted(requested_terms)} "
        f"against branding palette labels, reranked them using EmoSet-derived "
        f"brightness/colorfulness targets for {emotions}, explicit constraint penalties, "
        f"heuristic-derived palette constraints, and pairwise WCAG contrast scoring."
    )

    return {
        "query": {
            "emotions": emotions,
            "industry": industry,
            "style_keywords": style_keywords,
            "constraints": constraints,
        },
        "best_palette": best,
        "top_k_palettes": top,
        "recommended_text_colors": ["#FFFFFF", "#1A1A1A"],
        "rationale": rationale,
    }
