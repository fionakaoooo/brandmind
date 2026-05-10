# Revision Loop

## How It Works
1. Agent 2 generates draft brand kit
2. Agent 3 evaluates: WCAG + coherence + constraints
3. If all pass → approve and return
4. If any fail → generate specific feedback → back to Agent 2
5. Repeat up to 3 iterations
6. Return best draft if max iterations reached

## Feedback Format
Agent 3 returns specific actionable feedback:
- "Contrast ratio 2.8:1 below 4.5 minimum — regenerate palette"
- "Purple tones inconsistent with Organic archetype"
- "Neon color #19e3eb violates no-neon constraint"

## Self-Improving Heuristics
Rule weights updated after each QC pass:
- Rules associated with higher QC scores → weight increased
- Rules associated with lower QC scores → weight decreased
