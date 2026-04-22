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


# 品牌情绪 -> EmoSet原生8类情绪 的桥接映射
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
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hsv_scaled(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if mx == 0 else diff / mx
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
        "avg_colorfulness": avg_s,  # saturation 作为 colorfulness proxy
    }


def _build_emoset_profile(emotions: List[str]) -> Dict[str, float]:
    """
    先直接匹配 emotion；
    如果匹配不到，再走 品牌情绪 -> EmoSet 原生情绪 映射。
    """
    df = load_emoset_summary()
    if df.empty:
        return {"brightness_target": 0.55, "colorfulness_target": 0.55}

    emotions_norm = [_normalize_col(e) for e in emotions]

    # 1) 直接匹配
    direct_hit = df[df["emotion"].astype(str).map(_normalize_col).isin(set(emotions_norm))]
    if not direct_hit.empty:
        return {
            "brightness_target": float(direct_hit["brightness_mean"].mean()),
            "colorfulness_target": float(direct_hit["colorfulness_mean"].mean()),
        }

    # 2) 映射到 EmoSet 原生情绪
    mapped = []
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
    penalty = 0.0
    constraint_text = " ".join(constraints).lower()

    for h in hex_codes:
        hue, sat, val = _rgb_to_hsv_scaled(_hex_to_rgb(h))

        # no red
        if "no red" in constraint_text:
            if hue < 25 or hue > 335:
                penalty += 1.2

        # avoid harsh colors
        if "avoid harsh colors" in constraint_text:
            if sat > 0.60:
                penalty += 1.2
            if val > 0.85 and sat > 0.45:
                penalty += 0.8
            if hue < 25 or hue > 335:
                penalty += 1.0

        # soft / calm
        if "soft" in constraint_text or "calm" in constraint_text:
            if sat > 0.60:
                penalty += 0.8

        # luxury / premium
        if "luxury" in constraint_text or "premium" in constraint_text:
            if sat > 0.65:
                penalty += 0.6

    # palette-level penalties
    if "avoid harsh colors" in constraint_text:
        if stats["avg_colorfulness"] > 0.55:
            penalty += 2.0
        if stats["avg_brightness"] > 0.82 and stats["avg_colorfulness"] > 0.45:
            penalty += 1.0

    if "soft" in constraint_text or "calm" in constraint_text:
        if stats["avg_colorfulness"] > 0.50:
            penalty += 1.2

    return penalty


def color_retrieve(
    emotions: List[str],
    industry: str = "",
    style_keywords: List[str] | None = None,
    constraints: List[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    从 branding palette dataset 检索候选 palette，
    再结合 EmoSet visual profile 与 constraint penalty 进行重排。
    """
    style_keywords = style_keywords or []
    constraints = constraints or []

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

    scored_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_labels = [c for c in label_cols if _to_binary(row[c]) == 1]
        row_label_set = set(row_labels)

        emotion_hits = len(requested_terms & row_label_set)
        base_score = float(emotion_hits) * 3.0
        industry_score = _industry_bonus(row, industry)

        hex_codes = [str(row[c]).strip() for c in hex_cols]
        stats = _palette_stats(hex_codes)

        brightness_gap = _distance(stats["avg_brightness"], emoset_profile["brightness_target"])
        colorfulness_gap = _distance(stats["avg_colorfulness"], emoset_profile["colorfulness_target"])

        emoset_alignment_score = max(
            0.0,
            2.0 - (brightness_gap + colorfulness_gap) * 2.5
        )

        penalty = _constraint_penalty(
            hex_codes=hex_codes,
            stats=stats,
            constraints=constraints,
        )

        total_score = base_score + industry_score + emoset_alignment_score - (penalty * 2.0)

        # 对 harsh palette 做更强约束
        constraint_text = " ".join(constraints).lower()
        if "avoid harsh colors" in constraint_text and penalty >= 2.5:
            total_score -= 4.0

        scored_rows.append({
            "palette_name": row.get("palette_name", row.get("name", f"palette_{len(scored_rows)+1}")),
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
            "penalty": round(penalty, 3),
            "total_score": round(total_score, 3),
        })

    scored_rows.sort(key=lambda x: x["total_score"], reverse=True)
    top = scored_rows[:top_k]
    best = top[0] if top else {}

    rationale = (
        f"Retrieved palettes by matching requested emotions/styles {sorted(requested_terms)} "
        f"against the branding palette labels, reranked them using EmoSet-derived "
        f"brightness/colorfulness targets for {emotions}, and applied constraint penalties "
        f"for {constraints if constraints else ['none']}."
    )

    # add WCAG anchor colors so palette always has high-contrast pairs
    if best and "hex_codes" in best:
        if "#FFFFFF" not in best["hex_codes"]:
            best["hex_codes"].append("#FFFFFF")
        if "#1A1A1A" not in best["hex_codes"]:
            best["hex_codes"].append("#1A1A1A")
    for p in top:
        if "hex_codes" in p:
            if "#FFFFFF" not in p["hex_codes"]:
                p["hex_codes"].append("#FFFFFF")
            if "#1A1A1A" not in p["hex_codes"]:
                p["hex_codes"].append("#1A1A1A")

    return {
        "query": {
            "emotions": emotions,
            "industry": industry,
            "style_keywords": style_keywords,
            "constraints": constraints,
        },
        "best_palette": best,
        "top_k_palettes": top,
        "rationale": rationale,
    }
