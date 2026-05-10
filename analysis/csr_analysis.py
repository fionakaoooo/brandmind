"""
Constraint satisfaction rate analysis
"""
import json

def analyze_csr(filepath):
    data = json.load(open(filepath))
    
    print("CSR by System:")
    for system, metrics in data['summary'].items():
        csr = metrics['constraint_satisfaction_rate']
        bar = '█' * int(csr * 20)
        print(f"  {system:<35} {csr:.3f} {bar}")
    
    print("\nBrandMind CSR per case:")
    for r in data['details']['brandmind_full']:
        case = r['case_id'].replace('case_', '')
        passed = r['evaluation']['constraint_pass_count']
        total = r['evaluation']['constraint_total']
        csr = passed/total
        status = '✓' if csr >= 0.8 else '✗'
        print(f"  {status} {case:<30} {passed}/{total} ({csr:.2f})")

if __name__ == "__main__":
    analyze_csr("reports/baseline_benchmark_wcag_loose.json")
