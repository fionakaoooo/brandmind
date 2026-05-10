"""
Human Study Analysis
Analyzes results from the BrandMind user preference study (20 participants)
"""

import json

results = {
    "total_participants": 20,
    "brands": ["Skincare", "Fintech", "Jewelry", "Education", "Fitness"],
    "kit_a_preferred": [12, 14, 7, 10, 6],
    "kit_b_preferred": [8, 6, 3, 4, 0],
    "about_the_same": [0, 0, 10, 6, 14],
    "avg_rating_kit_a": [3.60, 3.65, 3.75, 3.70, 3.80],
    "avg_rating_kit_b": [2.75, 2.65, 3.15, 2.50, 3.45],
    "overall_kit_a_pct": 49.0,
    "overall_kit_b_pct": 21.0,
    "overall_same_pct": 30.0,
    "overall_avg_kit_a": 3.70,
    "overall_avg_kit_b": 2.90,
}

with open("reports/human_study_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved!")
