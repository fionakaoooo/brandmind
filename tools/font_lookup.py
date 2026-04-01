
def font_lookup(archetype: str):
    archetype = (archetype or "").lower()

    font_map = {
        "luxury": ["Playfair Display", "Didot", "Bodoni"],
        "tech": ["Inter", "Roboto", "SF Pro"],
        "playful": ["Poppins", "Baloo", "Comic Neue"],
        "corporate": ["Helvetica", "Arial", "IBM Plex Sans"],
        "minimal": ["Inter", "Helvetica", "Neue Haas Grotesk"],
        "bold": ["Oswald", "Montserrat", "Anton"],
        "organic": ["Lora", "Merriweather", "Cormorant"],
        "artisan": ["Libre Baskerville", "Georgia", "Cormorant"],
        "heritage": ["Garamond", "Baskerville", "Caslon"],
        "youthful": ["Poppins", "Nunito", "Quicksand"],
    }

    return font_map.get(archetype, ["Inter", "Helvetica"])
