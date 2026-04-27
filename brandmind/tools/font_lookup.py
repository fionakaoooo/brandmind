from __future__ import annotations

import os
import requests
from functools import lru_cache
from typing import Any, Dict, List

GOOGLE_FONTS_API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"


def _get_google_fonts_api_key() -> str:
    return (
        os.environ.get("GOOGLE_FONTS_API_KEY", "").strip()
        or os.environ.get("GOOGLE_FONTS_KEY", "").strip()
    )
    
# Google Fonts 的类别包括: "sans-serif", "serif", "display", "handwriting", "monospace"
ARCHETYPE_CATEGORY_MAP = {
    "luxury": ["serif", "sans-serif"],
    "playful": ["display", "handwriting", "sans-serif"],
    "corporate": ["sans-serif", "serif"],
    "minimal": ["sans-serif"],
    "bold": ["display", "sans-serif"],
    "organic": ["handwriting", "serif", "sans-serif"],
    "tech": ["sans-serif", "monospace"],
    "artisan": ["serif", "handwriting"],
    "heritage": ["serif", "display"],
    "youthful": ["display", "sans-serif"],
    "neutral": ["sans-serif", "serif"] # 默认后备
}

@lru_cache(maxsize=1)
def fetch_all_google_fonts() -> List[Dict[str, Any]]:
    """
    获取并缓存所有的 Google Fonts 元数据。
    按流行度 (popularity) 排序，这样可以优先推荐高质量且常用的字体。
    """
    api_key = _get_google_fonts_api_key()
    if not api_key:
        print("Warning: GOOGLE_FONTS_API_KEY/GOOGLE_FONTS_KEY is not set. Using a fallback mock list.")
        return _fallback_mock_fonts()

    params = {
        "key": api_key,
        "sort": "popularity" # 极其重要：按流行度排序保证字体质量
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
    """如果 API 调用失败或未配置 Key，提供一些经典的后备选项，防止 Pipeline 崩溃"""
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
    """
    给候选字体打分。
    分数基于：类别匹配度、是否包含关键字、以及粗细变体(variants)的丰富度。
    """
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
    # 例如：如果 style 包含 "rounded" 且字体名包含 "round"
    if "round" in style_lower and "round" in family:
        score += 3.0
    if "condensed" in style_lower and "condensed" in family:
        score += 3.0
    if "slab" in style_lower and "slab" in family:
        score += 3.0

    # 3. 字体家族丰富度 (Versatility Bonus)
    # 变体越多的字体，越适合作为品牌字体（可以涵盖正文、标题等不同字重）
    if len(variants) >= 5:
        score += 2.0
    elif len(variants) >= 3:
        score += 1.0
        
    # 4. 惩罚项
    # 如果是纯大写或特殊 Display 字体，变体通常很少。如果系统需要高对比度，适当保留。
    if category == "handwriting" and "professional" in style_lower:
        score -= 2.0

    return score


def font_lookup(archetype: str, style: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Agent 2 调用的主要工具函数。
    根据给定的原型和风格，检索并返回最适合的字体候选集。
    """
    archetype_key = archetype.strip().lower()
    target_categories = ARCHETYPE_CATEGORY_MAP.get(archetype_key, ARCHETYPE_CATEGORY_MAP["neutral"])
    
    all_fonts = fetch_all_google_fonts()
    
    scored_fonts = []
    # 为了保证性能，我们只在 Google Fonts 流行度排名前 300 的字体中进行筛选，保证推荐的都是成熟、支持度高的字体
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
            
    # 按分数降序排序
    scored_fonts.sort(key=lambda x: x["score"], reverse=True)
    
    # 如果没有找到足够的高分候选，可以补充一些符合基本 category 的流行字体
    if len(scored_fonts) < top_k:
        fallback_added = 0
        for font in all_fonts:
            if font.get("category") in target_categories and not any(f["family"] == font.get("family") for f in scored_fonts):
                scored_fonts.append({
                    "family": font.get("family"),
                    "category": font.get("category"),
                    "variants": font.get("variants", []),
                    "subsets": font.get("subsets", []),
                    "score": 0.5 # 垫底分数
                })
                fallback_added += 1
                if fallback_added >= top_k:
                    break

    return scored_fonts[:top_k]

# ==========================================
# 测试代码 (可独立运行)
# ==========================================
if __name__ == "__main__":
    print("Testing Font Lookup for a 'Luxury' brand with 'high contrast serif' style...")
    results = font_lookup(archetype="luxury", style="high contrast serif", top_k=5)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['family']} ({res['category']}) - Score: {res['score']} - Variants: {len(res['variants'])}")
