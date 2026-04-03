"""
tools/wcag_check.py
Programmatic WCAG 2.1 AA contrast checker.
No LLM needed — pure math.
"""

from typing import List, Dict


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.strip().lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _relative_luminance(rgb) -> float:
    def channel(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = [channel(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = _relative_luminance(_hex_to_rgb(hex1))
    l2 = _relative_luminance(_hex_to_rgb(hex2))
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def evaluate_palette_wcag(hex_codes: List[str], level: str = "AA", large_text: bool = False) -> Dict:
    """
    Checks all foreground/background pairs in a palette for WCAG 2.1 AA compliance.
    AA requires contrast ratio >= 4.5 for normal text.
    Returns summary dict with pass/fail per pair and overall result.
    """
    results = []
    pairs_checked = 0
    pairs_passed = 0

    for i in range(len(hex_codes)):
        for j in range(len(hex_codes)):
            if i == j:
                continue
            ratio = contrast_ratio(hex_codes[i], hex_codes[j])
            passed = ratio >= 4.5
            pairs_checked += 1
            if passed:
                pairs_passed += 1
            results.append({
                "fg": hex_codes[i],
                "bg": hex_codes[j],
                "ratio": round(ratio, 2),
                "passes_AA": passed,
            })

    overall_pass = pairs_passed > 0
    return {
        "pairs_checked": pairs_checked,
        "pairs_passed": pairs_passed,
        "overall_pass": overall_pass,
        "details": results,
        "summary": f"{pairs_passed}/{pairs_checked} pairs pass WCAG 2.1 AA (ratio >= 4.5)",
    }


if __name__ == "__main__":
    test_palette = ["#1A1A2E", "#C9A96E", "#F5F0E8", "#8B7355", "#2C2C2C"]
    result = evaluate_palette_wcag(test_palette)
    print(result["summary"])
    for p in result["details"]:
        if p["passes_AA"]:
            print(f"  ✓ {p['fg']} on {p['bg']} = {p['ratio']}")
