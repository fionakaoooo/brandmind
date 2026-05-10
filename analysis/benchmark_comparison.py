"""
Compare benchmark results across different runs
"""
import json
import os

def compare_benchmarks():
    files = {
        'OpenAI 6-case': 'reports/baseline_benchmark_openai.json',
        'OpenAI 20-case loose WCAG': 'reports/baseline_benchmark_wcag_loose.json',
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        print(f"\n=== {name} ({data.get('cases_count', 'N/A')} cases) ===")
        for system, metrics in data.get('summary', {}).items():
            print(f"  {system:<35} CSR={metrics.get('constraint_satisfaction_rate', 0):.3f} WCAG={metrics.get('wcag_pass_rate', 0):.3f} Coh={metrics.get('archetype_coherence_score_mean', 0):.3f}")

if __name__ == "__main__":
    compare_benchmarks()
