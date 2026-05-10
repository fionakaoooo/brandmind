"""
Analyzes BrandMind performance by archetype
"""
import json

def analyze_by_archetype(filepath):
    data = json.load(open(filepath))
    runs = data['details']['brandmind_full']
    
    by_arch = {}
    for r in runs:
        arch = r['output']['archetype']
        if arch not in by_arch:
            by_arch[arch] = {'csr': [], 'coherence': [], 'wcag': []}
        by_arch[arch]['csr'].append(
            r['evaluation']['constraint_pass_count']/r['evaluation']['constraint_total'])
        by_arch[arch]['coherence'].append(r['evaluation']['coherence_score'])
        by_arch[arch]['wcag'].append(1 if r['evaluation']['wcag_all_pass'] else 0)
    
    print("Archetype Performance Summary:")
    print(f"{'Archetype':<15} {'CSR':>6} {'Coh':>6} {'WCAG':>6} {'N':>4}")
    print("-" * 40)
    for arch in sorted(by_arch.keys()):
        m = by_arch[arch]
        n = len(m['csr'])
        print(f"{arch:<15} {sum(m['csr'])/n:>6.3f} {sum(m['coherence'])/n:>6.2f} {sum(m['wcag'])/n:>6.2f} {n:>4}")

if __name__ == "__main__":
    analyze_by_archetype("reports/baseline_benchmark_wcag_loose.json")
