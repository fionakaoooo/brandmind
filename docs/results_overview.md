# BrandMind Results Overview

## Automated Benchmark (20 cases, 4 systems)

| System | CSR | WCAG | Coherence | Runtime |
|---|---|---|---|---|
| BrandMind (ours) | 0.568 | **0.950** | 4.320 | 37.2s |
| Zero-shot GPT-4o | 0.622 | 0.600 | **4.391** | 3.9s |
| RAG only | 0.662 | 0.750 | 3.895 | 2.6s |
| Fontjoy | 0.095 | 0.000 | 2.415 | 1.4s |

### Key Findings
- BrandMind achieves highest WCAG compliance (0.95) — anchor colors strategy works
- BrandMind and Zero-shot tied on coherence (~4.3/5)
- Fontjoy completely fails on WCAG and CSR — confirms rule-based tools are insufficient

## Human Study (20 participants, 5 brands)

| Metric | Result |
|---|---|
| Kit A (BrandMind) preferred | 49% |
| Kit B (Zero-shot) preferred | 21% |
| About the same | 30% |
| Avg Kit A rating | 3.70/5 |
| Avg Kit B rating | 2.90/5 |
| Rating gap | +0.80 in favor of BrandMind |

### Strongest result
Fintech brand: 14/20 participants preferred BrandMind

### Most neutral result  
Fitness brand: 14/20 said "About the same"
