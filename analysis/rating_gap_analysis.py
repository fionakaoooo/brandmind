"""
Analyzes the rating gap between Kit A and Kit B in human study
"""

brands = ['Skincare', 'Fintech', 'Jewelry', 'Education', 'Fitness']
kit_a = [3.60, 3.65, 3.75, 3.70, 3.80]
kit_b = [2.75, 2.65, 3.15, 2.50, 3.45]

print("Rating Gap Analysis (Kit A - Kit B):")
print(f"{'Brand':<12} {'Kit A':>6} {'Kit B':>6} {'Gap':>6}")
print("-" * 35)

gaps = []
for brand, a, b in zip(brands, kit_a, kit_b):
    gap = a - b
    gaps.append(gap)
    print(f"{brand:<12} {a:>6.2f} {b:>6.2f} {gap:>+6.2f}")

print("-" * 35)
print(f"{'Overall':<12} {sum(kit_a)/5:>6.2f} {sum(kit_b)/5:>6.2f} {sum(gaps)/5:>+6.2f}")
print()
print(f"Max gap: {max(gaps):.2f} ({brands[gaps.index(max(gaps))]})")
print(f"Min gap: {min(gaps):.2f} ({brands[gaps.index(min(gaps))]})")
print(f"All gaps positive: {all(g > 0 for g in gaps)}")
