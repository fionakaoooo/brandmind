"""
Identifies top and bottom performing cases
"""
import json

data = json.load(open('reports/baseline_benchmark_wcag_loose.json'))
runs = data['details']['brandmind_full']

scored = []
for r in runs:
    csr = r['evaluation']['constraint_pass_count']/r['evaluation']['constraint_total']
    coh = r['evaluation']['coherence_score']
    wcag = 1 if r['evaluation']['wcag_all_pass'] else 0
    total = (csr + wcag + coh/5) / 3
    scored.append({
        'case': r['case_id'],
        'archetype': r['output']['archetype'],
        'csr': round(csr, 3),
        'coherence': coh,
        'wcag': wcag,
        'total': round(total, 3),
    })

scored.sort(key=lambda x: -x['total'])

print("Top 5 Cases:")
for s in scored[:5]:
    print(f"  {s['case']}: total={s['total']} csr={s['csr']} coh={s['coherence']} wcag={s['wcag']}")

print("\nBottom 5 Cases:")
for s in scored[-5:]:
    print(f"  {s['case']}: total={s['total']} csr={s['csr']} coh={s['coherence']} wcag={s['wcag']}")
