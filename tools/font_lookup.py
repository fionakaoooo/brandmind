def font_lookup(archetype: str, style: str = "", top_k: int = 5):
    archetype = (archetype or "").lower()
    style = (style or "").lower()

    font_db = {
        "luxury": [
            {"family": "Playfair Display", "category": "serif"},
            {"family": "Cormorant Garamond", "category": "serif"},
            {"family": "Bodoni Moda", "category": "serif"},
            {"family": "Inter", "category": "sans-serif"},
            {"family": "Helvetica", "category": "sans-serif"},
        ],
        "tech": [
            {"family": "Inter", "category": "sans-serif"},
            {"family": "IBM Plex Sans", "category": "sans-serif"},
            {"family": "Roboto", "category": "sans-serif"},
            {"family": "Space Grotesk", "category": "sans-serif"},
            {"family": "SF Pro", "category": "sans-serif"},
        ],
        "playful": [
            {"family": "Poppins", "category": "sans-serif"},
            {"family": "Nunito", "category": "sans-serif"},
            {"family": "Quicksand", "category": "sans-serif"},
            {"family": "Baloo 2", "category": "display"},
            {"family": "Fredoka", "category": "display"},
        ],
        "corporate": [
            {"family": "Helvetica", "category": "sans-serif"},
            {"family": "Arial", "category": "sans-serif"},
            {"family": "IBM Plex Sans", "category": "sans-serif"},
            {"family": "Inter", "category": "sans-serif"},
            {"family": "Source Sans 3", "category": "sans-serif"},
        ],
        "minimal": [
            {"family": "Inter", "category": "sans-serif"},
            {"family": "Helvetica", "category": "sans-serif"},
            {"family": "Neue Haas Grotesk", "category": "sans-serif"},
            {"family": "Manrope", "category": "sans-serif"},
            {"family": "DM Sans", "category": "sans-serif"},
        ],
        "bold": [
            {"family": "Oswald", "category": "display"},
            {"family": "Anton", "category": "display"},
            {"family": "Montserrat", "category": "sans-serif"},
            {"family": "Bebas Neue", "category": "display"},
            {"family": "Archivo Black", "category": "display"},
        ],
        "organic": [
            {"family": "Lora", "category": "serif"},
            {"family": "Merriweather", "category": "serif"},
            {"family": "Cormorant Garamond", "category": "serif"},
            {"family": "Nunito Sans", "category": "sans-serif"},
            {"family": "Source Serif 4", "category": "serif"},
        ],
        "artisan": [
            {"family": "Libre Baskerville", "category": "serif"},
            {"family": "Cormorant Garamond", "category": "serif"},
            {"family": "Lora", "category": "serif"},
            {"family": "Figtree", "category": "sans-serif"},
            {"family": "Alegreya", "category": "serif"},
        ],
        "heritage": [
            {"family": "Garamond", "category": "serif"},
            {"family": "Baskerville", "category": "serif"},
            {"family": "Libre Baskerville", "category": "serif"},
            {"family": "Cormorant Garamond", "category": "serif"},
            {"family": "Georgia", "category": "serif"},
        ],
        "youthful": [
            {"family": "Poppins", "category": "sans-serif"},
            {"family": "Nunito", "category": "sans-serif"},
            {"family": "Quicksand", "category": "sans-serif"},
            {"family": "Fredoka", "category": "display"},
            {"family": "Montserrat", "category": "sans-serif"},
        ],
    }

    candidates = font_db.get(archetype, [
        {"family": "Inter", "category": "sans-serif"},
        {"family": "Helvetica", "category": "sans-serif"},
        {"family": "Lora", "category": "serif"},
        {"family": "Montserrat", "category": "sans-serif"},
        {"family": "Playfair Display", "category": "serif"},
    ])

    # 简单 style 偏置
    if "serif" in style:
        serif_first = [f for f in candidates if "serif" in f["category"]]
        non_serif = [f for f in candidates if "serif" not in f["category"]]
        candidates = serif_first + non_serif
    elif "sans" in style:
        sans_first = [f for f in candidates if "sans" in f["category"]]
        non_sans = [f for f in candidates if "sans" not in f["category"]]
        candidates = sans_first + non_sans
    elif "display" in style:
        display_first = [f for f in candidates if f["category"] == "display"]
        others = [f for f in candidates if f["category"] != "display"]
        candidates = display_first + others

    return candidates[:top_k]
