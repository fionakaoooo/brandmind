# WCAG Implementation Notes

## Standard
WCAG 2.1 Level AA — contrast ratio >= 4.5 for normal text

## Implementation
- All palettes anchored with #FFFFFF and #1A1A1A
- Contrast computed for all k(k-1) foreground/background pairs
- Palette passes if at least 1 pair meets threshold (min-pairs mode)

## Tools
- tools/wcag_check.py — evaluate_palette_wcag()
- tools/wcag_check.py — evaluate_palette_wcag_min_pairs()

## Results
BrandMind WCAG pass rate: 0.95 (20 cases)
Zero-shot WCAG pass rate: 0.60
RAG-only WCAG pass rate: 0.75
Fontjoy WCAG pass rate: 0.00
