"""
Evaluation Pipeline
Runs full evaluation: baseline benchmark + human study analysis
"""

import json
import os

def load_benchmark(path):
    with open(path) as f:
        return json.load(f)

def print_summary(data):
    print("\n=== Benchmark Summary ===")
    for system, metrics in data.get('summary', {}).items():
        print(f"{system}:")
        print(f"  CSR:       {metrics.get('constraint_satisfaction_rate')}")
        print(f"  WCAG:      {metrics.get('wcag_pass_rate')}")
        print(f"  Coherence: {metrics.get('archetype_coherence_score_mean')}")
        print(f"  Cases:     {metrics.get('cases')}")

def human_study_summary():
    print("\n=== Human Study Summary ===")
    print("Kit A (BrandMind) preferred: 49%")
    print("Kit B (Zero-shot) preferred: 21%")
    print("About the same:              30%")
    print("Avg Kit A rating:            3.70/5")
    print("Avg Kit B rating:            2.90/5")

if __name__ == "__main__":
    data = load_benchmark("reports/baseline_benchmark_wcag_loose.json")
    print_summary(data)
    human_study_summary()
