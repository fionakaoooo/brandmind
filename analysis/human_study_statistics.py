"""
Statistical analysis of human study results
"""

data = {
    'skincare':  {'kit_a': 12, 'kit_b': 8,  'same': 0,  'avg_a': 3.60, 'avg_b': 2.75},
    'fintech':   {'kit_a': 14, 'kit_b': 6,  'same': 0,  'avg_a': 3.65, 'avg_b': 2.65},
    'jewelry':   {'kit_a': 7,  'kit_b': 3,  'same': 10, 'avg_a': 3.75, 'avg_b': 3.15},
    'education': {'kit_a': 10, 'kit_b': 4,  'same': 6,  'avg_a': 3.70, 'avg_b': 2.50},
    'fitness':   {'kit_a': 6,  'kit_b': 0,  'same': 14, 'avg_a': 3.80, 'avg_b': 3.45},
}

total_a = sum(d['kit_a'] for d in data.values())
total_b = sum(d['kit_b'] for d in data.values())
total_same = sum(d['same'] for d in data.values())
total = total_a + total_b + total_same

print("=== Human Study Statistical Summary ===")
print(f"Total judgments: {total}")
print(f"Kit A (BrandMind): {total_a} ({total_a/total*100:.1f}%)")
print(f"Kit B (Zero-shot): {total_b} ({total_b/total*100:.1f}%)")
print(f"About the same:    {total_same} ({total_same/total*100:.1f}%)")
print()
print("Per brand rating gap (Kit A - Kit B):")
for brand, d in data.items():
    gap = d['avg_a'] - d['avg_b']
    print(f"  {brand:<12}: +{gap:.2f} in favor of BrandMind")
print()
overall_gap = sum(d['avg_a'] for d in data.values())/5 - sum(d['avg_b'] for d in data.values())/5
print(f"Overall rating gap: +{overall_gap:.2f}")
