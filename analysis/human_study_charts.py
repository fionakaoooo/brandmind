"""
Generate all human study charts
"""
import matplotlib.pyplot as plt
import numpy as np

brands = ['Skincare', 'Fintech', 'Jewelry', 'Education', 'Fitness']
kit_a_pref = [12, 14, 7, 10, 6]
kit_b_pref = [8, 6, 3, 4, 0]
same_pref  = [0, 0, 10, 6, 14]
kit_a_avg  = [3.60, 3.65, 3.75, 3.70, 3.80]
kit_b_avg  = [2.75, 2.65, 3.15, 2.50, 3.45]

x = np.arange(len(brands))
width = 0.25

# preference chart
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width, kit_a_pref, width, label='Kit A (BrandMind)', color='#4C9BE8')
ax.bar(x,         same_pref,  width, label='About the same',    color='#A8A8A8')
ax.bar(x + width, kit_b_pref, width, label='Kit B (Zero-shot)', color='#E8874C')
ax.set_xlabel('Brand')
ax.set_ylabel('Number of Participants (out of 20)')
ax.set_title('Human Study: Kit Preference by Brand (n=20)')
ax.set_xticks(x)
ax.set_xticklabels(brands)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig('reports/human_study_preference.png', dpi=150)
plt.close()

# ratings chart
fig, ax = plt.subplots(figsize=(11, 5))
bars_a = ax.bar(x - 0.2, kit_a_avg, 0.35, label='Kit A (BrandMind)', color='#4C9BE8')
bars_b = ax.bar(x + 0.2, kit_b_avg, 0.35, label='Kit B (Zero-shot)', color='#E8874C')
ax.set_xlabel('Brand')
ax.set_ylabel('Average Rating (1-5)')
ax.set_title('Human Study: Average Brand Fit Ratings (n=20)')
ax.set_xticks(x)
ax.set_xticklabels(brands)
ax.legend()
ax.set_ylim(0, 5)
for bar, val in zip(bars_a, kit_a_avg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f'{val:.2f}', ha='center', fontsize=8)
for bar, val in zip(bars_b, kit_b_avg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f'{val:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('reports/human_study_ratings.png', dpi=150)
plt.close()
print('All human study charts saved!')
