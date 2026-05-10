"""
Analyzes font choices made by BrandMind across 20 test cases
"""
import json

data = json.load(open('reports/baseline_benchmark_wcag_loose.json'))
runs = data['details']['brandmind_full']

headline_fonts = {}
body_fonts = {}
categories = {}

for r in runs:
    kit = r['output'].get('kit', {})
    font_rec = kit.get('font_recommendation', {})
    
    headline = font_rec.get('headline', {})
    body = font_rec.get('body', {})
    
    if isinstance(headline, dict):
        hf = headline.get('family', 'unknown')
        hc = headline.get('category', 'unknown')
        headline_fonts[hf] = headline_fonts.get(hf, 0) + 1
        categories[hc] = categories.get(hc, 0) + 1
    
    if isinstance(body, dict):
        bf = body.get('family', 'unknown')
        body_fonts[bf] = body_fonts.get(bf, 0) + 1

print("Top Headline Fonts:")
for f, c in sorted(headline_fonts.items(), key=lambda x: -x[1]):
    print(f"  {f}: {c}")

print("\nTop Body Fonts:")
for f, c in sorted(body_fonts.items(), key=lambda x: -x[1]):
    print(f"  {f}: {c}")

print("\nFont Categories:")
for c, n in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")
