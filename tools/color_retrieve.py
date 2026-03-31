from __future__ import annotations

import json
import math
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
        # soft fallback
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

    # common fallback names
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
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hsv_scaled(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn

    # hue
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

    # simple proxy: colorfulness ~= saturation
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

    emotions_norm = {_normalize_col(e) for e in emotions}
    hit = df[df["emotion"].astype(str).map(_normalize_col).isin(emotions_norm)]

    if hit.empty:
        return {"brightness_target": 0.55, "colorfulness_target": 0.55}

    return {
        "brightness_target": float(hit["brightness_mean"].mean()),
        "colorfulness_target": float(hit["colorfulness_mean"].mean()),
    }


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


def color_retrieve(
    emotions: List[str],
    industry: str = "",
    style_keywords: List[str] | None = None,
    constraints: List[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve candidate palettes from the branding palette dataset and rerank them
    with EmoSet-based visual grounding.
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

        brightness_gap = _distance(
            stats["avg_brightness"], emoset_profile["brightness_target"]
        )
        colorfulness_gap = _distance(
            stats["avg_colorfulness"], emoset_profile["colorfulness_target"]
        )

        emoset_alignment_score = max(0.0, 2.0 - (brightness_gap + colorfulness_gap) * 2.5)

        penalty = 0.0
        constraint_text = " ".join(constraints).lower()
        if "no red" in constraint_text:
            # rough heuristic: penalize if too many reds/oranges
            if any(_rgb_to_hsv_scaled(_hex_to_rgb(h))[0] < 25 or _rgb_to_hsv_scaled(_hex_to_rgb(h))[0] > 335 for h in hex_codes):
                penalty += 1.5

        total_score = base_score + industry_score + emoset_alignment_score - penalty

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
        f"against the branding palette labels, then reranked them using EmoSet-derived "
        f"brightness/colorfulness targets for {emotions}."
    )

    return {
        "query": {
            "emotions": emotions,
            "industry": industry,
            "style_keywords": style_keywords,
        },
        "best_palette": best,
        "top_k_palettes": top,
        "rationale": rationale,
    }
