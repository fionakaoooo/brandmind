from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _normalize_hex(color: str) -> str:
    color = str(color).strip()
    if not color.startswith("#"):
        color = f"#{color}"
    return color.upper()


def _is_valid_hex(color: str) -> bool:
    return bool(HEX_RE.match(_normalize_hex(color)))


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = _normalize_hex(color).lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def _srgb_to_linear(channel: float) -> float:
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    r, g, b = _hex_to_rgb(color)
    r_lin = _srgb_to_linear(r / 255.0)
    g_lin = _srgb_to_linear(g / 255.0)
    b_lin = _srgb_to_linear(b / 255.0)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def contrast_ratio(fg: str, bg: str) -> float:
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    lighter, darker = (l1, l2) if l1 >= l2 else (l2, l1)
    return (lighter + 0.05) / (darker + 0.05)


def wcag_contrast_check(
    fg: str,
    bg: str,
    level: str = "AA",
    large_text: bool = False,
) -> Dict[str, Any]:
    fg_norm = _normalize_hex(fg)
    bg_norm = _normalize_hex(bg)

    if not _is_valid_hex(fg_norm) or not _is_valid_hex(bg_norm):
        return {
            "fg": fg_norm,
            "bg": bg_norm,
            "ratio": 0.0,
            "threshold": None,
            "passes": False,
            "error": "Invalid hex color input.",
        }

    level_norm = str(level).strip().upper()
    if level_norm not in {"AA", "AAA"}:
        level_norm = "AA"

    if level_norm == "AA":
        threshold = 3.0 if large_text else 4.5
    else:
        threshold = 4.5 if large_text else 7.0

    ratio = contrast_ratio(fg_norm, bg_norm)

    return {
        "fg": fg_norm,
        "bg": bg_norm,
        "ratio": round(ratio, 3),
        "threshold": threshold,
        "passes": ratio >= threshold,
        "level": level_norm,
        "large_text": bool(large_text),
    }


def evaluate_palette_wcag(
    hex_codes: List[str],
    level: str = "AA",
    large_text: bool = False,
) -> Dict[str, Any]:
    normalized = [_normalize_hex(h) for h in (hex_codes or [])]
    valid = [h for h in normalized if _is_valid_hex(h)]

    if not valid:
        return {
            "all_pass": False,
            "pass_rate": 0.0,
            "pass_count": 0,
            "total_checks": 0,
            "color_checks": [],
            "pair_checks": [],
            "invalid_colors": normalized,
            "message": "No valid hex codes found in palette.",
        }

    invalid_colors = [h for h in normalized if h not in valid]

    color_checks: List[Dict[str, Any]] = []
    pass_count = 0
    for bg in valid:
        white = wcag_contrast_check("#FFFFFF", bg, level=level, large_text=large_text)
        black = wcag_contrast_check("#111111", bg, level=level, large_text=large_text)
        best = white if white["ratio"] >= black["ratio"] else black

        color_result = {
            "background": bg,
            "best_text_color": best["fg"],
            "best_ratio": best["ratio"],
            "threshold": best["threshold"],
            "passes": bool(best["passes"]),
            "white_ratio": white["ratio"],
            "black_ratio": black["ratio"],
        }
        color_checks.append(color_result)
        if color_result["passes"]:
            pass_count += 1

    pair_checks: List[Dict[str, Any]] = []
    threshold = color_checks[0]["threshold"] if color_checks else 4.5
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            fg = valid[i]
            bg = valid[j]
            ratio = round(contrast_ratio(fg, bg), 3)
            pair_checks.append(
                {
                    "fg": fg,
                    "bg": bg,
                    "ratio": ratio,
                    "threshold": threshold,
                    "passes": ratio >= threshold,
                }
            )

    total_checks = len(color_checks)
    pass_rate = (pass_count / total_checks) if total_checks else 0.0

    return {
        "all_pass": pass_count == total_checks and total_checks > 0,
        "pass_rate": round(pass_rate, 3),
        "pass_count": pass_count,
        "total_checks": total_checks,
        "color_checks": color_checks,
        "pair_checks": pair_checks,
        "invalid_colors": invalid_colors,
        "level": str(level).upper(),
        "large_text": bool(large_text),
    }


if __name__ == "__main__":
    sample = ["#1A1A1A", "#F4F4F4", "#4A7C59", "#C72C41", "#E6C229"]
    report = evaluate_palette_wcag(sample)
    print(report)
