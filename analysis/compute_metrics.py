"""
Compute final metrics from benchmark results
"""
import json

def compute_metrics(filepath):
    data = json.load(open(filepath))
    summary = data.get('summary', {})
    
    print(f"File: {filepath}")
    print(f"Cases: {data.get('cases_count', 'N/A')}")
    print()
    
    for system, metrics in summary.items():
        print(f"{system}:")
        print(f"  CSR:       {metrics.get('constraint_satisfaction_rate', 'N/A'):.3f}")
        print(f"  WCAG:      {metrics.get('wcag_pass_rate', 'N/A'):.3f}")
        print(f"  Coherence: {metrics.get('archetype_coherence_score_mean', 'N/A'):.3f}")
        print(f"  Runtime:   {metrics.get('avg_runtime_sec', 'N/A'):.1f}s")
        print()

if __name__ == "__main__":
    compute_metrics("reports/baseline_benchmark_wcag_loose.json")
