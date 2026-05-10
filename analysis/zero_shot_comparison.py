"""
Detailed BrandMind vs Zero-shot comparison per case
"""
import json

data = json.load(open('reports/baseline_benchmark_wcag_loose.json'))
bm = data['details']['brandmind_full']
zs = data['details']['baseline_zero_shot_gpt4o']

print(f"{'Case':<30} {'BM_CSR':>8} {'ZS_CSR':>8} {'BM_Coh':>8} {'ZS_Coh':>8} {'Winner':>8}")
print("-" * 75)

bm_wins = 0
zs_wins = 0
ties = 0

for b, z in zip(bm, zs):
    case = b['case_id'].replace('case_', '').replace('_', ' ')
    b_csr = b['evaluation']['constraint_pass_count']/b['evaluation']['constraint_total']
    z_csr = z['evaluation']['constraint_pass_count']/z['evaluation']['constraint_total']
    b_coh = b['evaluation']['coherence_score']
    z_coh = z['evaluation']['coherence_score']
    
    b_score = (b_csr + b_coh/5) / 2
    z_score = (z_csr + z_coh/5) / 2
    
    if b_score > z_score:
        winner = 'BrandMind'
        bm_wins += 1
    elif z_score > b_score:
        winner = 'Zero-shot'
        zs_wins += 1
    else:
        winner = 'Tie'
        ties += 1
    
    print(f"{case:<30} {b_csr:>8.3f} {z_csr:>8.3f} {b_coh:>8.2f} {z_coh:>8.2f} {winner:>8}")

print(f"\nBrandMind wins: {bm_wins}/20")
print(f"Zero-shot wins: {zs_wins}/20")
print(f"Ties: {ties}/20")
