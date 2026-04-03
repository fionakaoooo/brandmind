from __future__ import annotations

import os
import requests
from functools import lru_cache
from typing import Any, Dict, List

GOOGLE_FONTS_API_KEY = os.environ.get("GOOGLE_FONTS_API_KEY", "")
GOOGLE_FONTS_API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"

ARCHETYPE_CATEGORY_MAP = {
    "luxury": ["serif", "sans-serif"],
    "playful": ["display", "handwriting", "sans-serif"],
    "corporate": ["sans-serif", "serif"],
    "minimal": ["sans-serif"],
    "bold": ["display", "sans-serif"],
    "organic": ["handwriting", "serif", "sans-serif"],
    "tech": ["sans-serif", "monospace"],
    "artisan": ["serif", "handwriting"],
    "heritage": ["serif"],
    "youthful": ["display", "sans-serif"],
    "neutral": ["sans-serif", "serif"] # 默认后备
}

@lru_cache(maxsize=1)
def fetch_all_google_fonts() -> List[Dict[str, Any]]:
    """Fetch all fonts from Google Fonts API."""
    if not GOOGLE_FONTS_API_KEY:
        print("Warning: GOOGLE_FONTS_API_KEY is not set. Using a fallback mock list.")
        return _fallback_mock_fonts()

    params = {
        "key": GOOGLE_FONTS_API_KEY,
        "sort": "popularity" 
    }
    
    try:
        response = requests.get(GOOGLE_FONTS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])
    except Exception as e:
        print(f"Error fetching Google Fonts: {e}. Falling back to mock data.")
        return _fallback_mock_fonts()


def _fallback_mock_fonts() -> List[Dict[str, Any]]:
    return [
        {"family": "Roboto", "category": "sans-serif", "variants": ["regular", "700"]},
        {"family": "Playfair Display", "category": "serif", "variants": ["regular", "italic", "700"]},
        {"family": "Montserrat", "category": "sans-serif", "variants": ["300", "regular", "700"]},
        {"family": "Merriweather", "category": "serif", "variants": ["300", "regular", "700"]},
        {"family": "Oswald", "category": "sans-serif", "variants": ["regular", "700"]},
        {"family": "Pacifico", "category": "handwriting", "variants": ["regular"]},
        {"family": "Space Mono", "category": "monospace", "variants": ["regular", "700"]},
        {"family": "Bebas Neue", "category": "display", "variants": ["regular"]}
    ]


def _score_font(font: Dict[str, Any], archetype: str, style: str, target_categories: List[str]) -> float:
    score = 0.0
    category = font.get("category", "")
    family = font.get("family", "").lower()
    variants = font.get("variants", [])
    style_lower = style.lower()

    # 1. 类别匹配 (Category Matching)
    if category in target_categories:
        # 如果是该原型最首选的类别，给最高分
        if target_categories.index(category) == 0:
            score += 5.0
        else:
            score += 3.0
            
    # 2. 风格关键字匹配 (Style Keyword Matching)
    if "round" in style_lower and "round" in family:
        score += 3.0
    if "condensed" in style_lower and "condensed" in family:
        score += 3.0
    if "slab" in style_lower and "slab" in family:
        score += 3.0

    # 3. 字体家族丰富度 (Versatility Bonus)
    if len(variants) >= 5:
        score += 2.0
    elif len(variants) >= 3:
        score += 1.0
        
    # 4. 惩罚项
    if category == "handwriting" and "professional" in style_lower:
        score -= 2.0

    return score


def font_lookup(archetype: str, style: str, top_k: int = 8) -> List[Dict[str, Any]]:
    archetype_key = archetype.strip().lower()
    target_categories = ARCHETYPE_CATEGORY_MAP.get(archetype_key, ARCHETYPE_CATEGORY_MAP["neutral"])
    
    all_fonts = fetch_all_google_fonts()
    
    scored_fonts = []
    for font in all_fonts[:300]:
        score = _score_font(font, archetype_key, style, target_categories)
        if score > 0:
            scored_fonts.append({
                "family": font.get("family"),
                "category": font.get("category"),
                "variants": font.get("variants", []),
                "subsets": font.get("subsets", []),
                "score": score
            })
            

    scored_fonts.sort(key=lambda x: x["score"], reverse=True)
    
    if len(scored_fonts) < top_k:
        fallback_added = 0
        for font in all_fonts:
            if font.get("category") in target_categories and not any(f["family"] == font.get("family") for f in scored_fonts):
                scored_fonts.append({
                    "family": font.get("family"),
                    "category": font.get("category"),
                    "variants": font.get("variants", []),
                    "subsets": font.get("subsets", []),
                    "score": 0.5 
                })
                fallback_added += 1
                if fallback_added >= top_k:
                    break

    return scored_fonts[:top_k]

if __name__ == "__main__":
    print("Testing Font Lookup for a 'Luxury' brand with 'high contrast serif' style...")
    results = font_lookup(archetype="luxury", style="high contrast serif", top_k=5)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['family']} ({res['category']}) - Score: {res['score']} - Variants: {len(res['variants'])}")
