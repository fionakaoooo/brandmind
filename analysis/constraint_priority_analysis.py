"""
Analyzes constraint priority distribution across 20 benchmark cases
"""
import json

data = json.load(open('reports/baseline_benchmark_wcag_loose.json'))
runs = data['details']['brandmind_full']

priority_counts = {'high': 0, 'medium': 0, 'low': 0}
high_kw = ['wcag', 'accessible', 'colorblind']
medium_kw = ['neon', 'color', 'serif', 'sans', 'font']

for r in runs:
    for c in r['constraints']:
        cl = c.lower()
        if any(k in cl for k in high_kw):
            priority_counts['high'] += 1
        elif any(k in cl for k in medium_kw):
            priority_counts['medium'] += 1
        else:
            priority_counts['low'] += 1

total = sum(priority_counts.values())
print("Constraint Priority Distribution:")
for p, n in priority_counts.items():
    print(f"  {p}: {n} ({n/total*100:.1f}%)")
