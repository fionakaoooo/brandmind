
def heuristic_search(palettes, top_k=5):
    """
    简化版：直接按 total_score 排序
    """
    if not palettes:
        return []

    sorted_palettes = sorted(
        palettes,
        key=lambda x: x.get("total_score", 0),
        reverse=True
    )

    return sorted_palettes[:top_k]
