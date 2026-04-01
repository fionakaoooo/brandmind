from __future__ import annotations

import os
import re
import math
import json
import requests
import pandas as pd
from functools import lru_cache
from typing import Any, Dict, List, Tuple
from getpass import getpass
from openai import OpenAI

# ==========================================
# 1. Configuration & API Setup
# ==========================================
# Set paths to match your processed data locations
PALETTE_CSV_PATH = "data/processed/branding_palettes_processed.csv"
EMOSET_SUMMARY_PATH = "data/processed/emoset_emotion_summary.csv"

# HEX Validation Regex
HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{6})$")

# ==========================================
# 2. Utility Functions
# ==========================================

def _normalize_col(name: str) -> str:
    return (
        str(name).strip().lower()
        .replace("-", "_").replace(" ", "_").replace("/", "_")
    )

@lru_cache(maxsize=1)
def load_palette_df() -> pd.DataFrame:
    if not os.path.exists(PALETTE_CSV_PATH):
        # Create dummy data if file is missing for testing
        return pd.DataFrame({"palette_name": ["Default"], "color1": ["#FFFFFF"], "industry": ["general"]})
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

def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = str(hex_code).strip().lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hsv_scaled(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn
    h = 0
    if diff != 0:
        if mx == r: h = (60 * ((g - b) / diff) + 360) % 360
        elif mx == g: h = (60 * ((b - r) / diff) + 120) % 360
        else: h = (60 * ((r - g) / diff) + 240) % 360
    s = 0 if mx == 0 else diff / mx
    return h, s, mx

def _palette_stats(hex_codes: List[str]) -> Dict[str, float]:
    hsvs = [_rgb_to_hsv_scaled(_hex_to_rgb(h)) for h in hex_codes if HEX_RE.match(str(h))]
    if not hsvs:
        return {"avg_brightness": 0.5, "avg_colorfulness": 0.5}
    return {
        "avg_brightness": sum(v for _, _, v in hsvs) / len(hsvs),
        "avg_colorfulness": sum(s for _, s, _ in hsvs) / len(hsvs)
    }

# ==========================================
# 3. Core Tools (Heuristics & Color Retrieval)
# ==========================================

def heuristic_search(brand_attribute: str) -> List[Dict[str, str]]:
    """Retrieves professional design rules based on brand archetypes."""
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

def _constraint_penalty(hex_codes: List[str], stats: Dict[str, float], constraints: List[str]) -> float:
    penalty = 0.0
    text = " ".join(constraints).lower()
    for h in hex_codes:
        hue, sat, val = _rgb_to_hsv_scaled(_hex_to_rgb(h))
        if "avoid harsh colors" in text:
            if sat > 0.72: penalty += 0.9
            if val > 0.90 and sat > 0.55: penalty += 0.6
        if "soft" in text or "calm" in text:
            if sat > 0.65: penalty += 0.7
    return penalty

def color_retrieve(emotions: List[str], industry: str = "", constraints: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
    """Retrieves palettes based on Branding dataset labels and EmoSet visual grounding."""
    df = load_palette_df()
    emo_df = load_emoset_summary()
    constraints = constraints or []
    
    # 1. Build EmoSet Target Profile
    emotions_norm = {_normalize_col(e) for e in emotions}
    hit = emo_df[emo_df["emotion"].astype(str).map(_normalize_col).isin(emotions_norm)]
    target_b = float(hit["brightness_mean"].mean()) if not hit.empty else 0.55
    target_c = float(hit["colorfulness_mean"].mean()) if not hit.empty else 0.55

    scored_rows = []
    # Simplified column detection logic
    hex_cols = [c for c in df.columns if "color" in c][:5]
    label_cols = [c for c in df.columns if c not in hex_cols and c not in ["palette_name", "industry"]]

    for _, row in df.iterrows():
        # Label Match Score
        row_labels = {c for c in label_cols if str(row[c]) in ["1", "1.0", "True", "true"]}
        base_score = float(len(emotions_norm & row_labels)) * 3.0
        
        # EmoSet Alignment Score
        hex_codes = [str(row[c]) for c in hex_cols if pd.notna(row[c])]
        stats = _palette_stats(hex_codes)
        gap = abs(stats["avg_brightness"] - target_b) + abs(stats["avg_colorfulness"] - target_c)
        alignment_score = max(0.0, 2.0 - gap * 2.5)
        
        # Constraint Penalty
        penalty = _constraint_penalty(hex_codes, stats, constraints)
        
        total_score = base_score + alignment_score - (penalty * 1.25)
        scored_rows.append({
            "palette_name": row.get("palette_name", "Unknown"),
            "hex_codes": hex_codes,
            "total_score": round(total_score, 3),
            "emoset_alignment": {"target_b": target_b, "actual_b": stats["avg_brightness"]}
        })

    scored_rows.sort(key=lambda x: x["total_score"], reverse=True)
    return scored_rows[0] if scored_rows else {}

# ==========================================
# 4. Main Agent Workflow
# ==========================================

def design_agent(brand_brief: str, archetype: str, constraints: List[str]) -> Dict[str, Any]:
    print(f"\n🎨 Starting Design Agent for: {brand_brief}")
    
    # Step 1: Tool Call - Heuristic Search
    print("🔍 Fetching Expert Rules...")
    design_rules = heuristic_search(archetype)
    
    # Step 2: Tool Call - Color Retrieval (Using EmoSet grounding)
    # Mapping archetype to primary emotions for EmoSet lookup
    emotion_map = {
        "premium": ["elegant", "calm"],
        "playful": ["amusement", "excitement"],
        "organic": ["contentment", "calm"]
    }
    target_emotions = emotion_map.get(archetype, ["contentment"])
    
    print(f"🌈 Retrieving grounded colors for emotions: {target_emotions}...")
    palette_data = color_retrieve(target_emotions, constraints=constraints)
    
    # Final Result Construction
    brand_kit = {
        "brand_identity": {
            "brief": brand_brief,
            "archetype": archetype
        },
        "visual_system": {
            "color_palette": palette_data,
            "expert_design_rules": design_rules
        }
    }
    return brand_kit

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # Test Scenario
    result = design_agent(
        brand_brief="A sustainable luxury skincare brand",
        archetype="premium",
        constraints=["avoid harsh colors"]
    )
    
    print("\n✅ GENERATED BRAND KIT:")
    print(json.dumps(result, indent=2))
