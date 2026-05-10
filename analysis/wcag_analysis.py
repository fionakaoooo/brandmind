"""
Detailed WCAG compliance analysis
"""
import json

def analyze_wcag(filepath):
    data = json.load(open(filepath))
    
    print("WCAG Pass Rate by System:")
    for system, metrics in data['summary'].items():
        rate = metrics['wcag_pass_rate']
        bar = '█' * int(rate * 20)
        print(f"  {system:<35} {rate:.3f} {bar}")
    
    print("\nBrandMind WCAG per case:")
    for r in data['details']['brandmind_full']:
        case = r['case_id'].replace('case_', '')
        wcag = r['evaluation']['wcag_pass_rate']
        passed = '✓' if r['evaluation']['wcag_all_pass'] else '✗'
        print(f"  {passed} {case:<30} {wcag:.3f}")

if __name__ == "__main__":
    analyze_wcag("reports/baseline_benchmark_wcag_loose.json")
