"""
Coherence score analysis across systems and cases
"""
import json

def analyze_coherence(filepath):
    data = json.load(open(filepath))
    
    print("Coherence by System:")
    for system, metrics in data['summary'].items():
        coh = metrics['archetype_coherence_score_mean']
        bar = '█' * int(coh * 4)
        print(f"  {system:<35} {coh:.3f} {bar}")
    
    print("\nBrandMind vs Zero-shot coherence per case:")
    bm = data['details']['brandmind_full']
    zs = data['details']['baseline_zero_shot_gpt4o']
    for b, z in zip(bm, zs):
        case = b['case_id'].replace('case_', '')
        b_coh = b['evaluation']['coherence_score']
        z_coh = z['evaluation']['coherence_score']
        winner = 'BM' if b_coh > z_coh else 'ZS' if z_coh > b_coh else '=='
        print(f"  {winner} {case:<28} BM={b_coh:.1f} ZS={z_coh:.1f}")

if __name__ == "__main__":
    analyze_coherence("reports/baseline_benchmark_wcag_loose.json")
