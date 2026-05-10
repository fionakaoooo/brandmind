"""
Integration test for BrandMind pipeline
Tests two contrasting brand briefs end to end
"""
import sys
sys.path.insert(0, '.')
from graph import run_pipeline

TEST_BRIEFS = [
    {
        "name": "Verdant Skincare",
        "brief": "Sustainable skincare brand for eco-conscious women 25-40. Warm, premium, grounded in nature. WCAG accessible. No neon colors.",
        "expected_archetype": "Organic"
    },
    {
        "name": "Vault Fintech", 
        "brief": "Fintech startup for young professionals 22-35. Cutting-edge, minimal, trustworthy. No traditional bank vibes. WCAG accessible.",
        "expected_archetype": "Tech"
    }
]

def run_tests():
    results = []
    for test in TEST_BRIEFS:
        print(f"\nTesting: {test['name']}")
        result = run_pipeline(test['brief'])
        passed = result.get('archetype') == test['expected_archetype']
        results.append({
            "name": test['name'],
            "archetype_match": passed,
            "archetype": result.get('archetype'),
            "coherence": result.get('qc_scores', {}).get('coherence', {}).get('score'),
            "constraints": result.get('qc_scores', {}).get('constraints', {}).get('pass_rate'),
        })
        print(f"  Archetype: {result.get('archetype')} ({'✓' if passed else '✗'})")
        print(f"  Coherence: {result.get('qc_scores', {}).get('coherence', {}).get('score')}")
        print(f"  Constraints: {result.get('qc_scores', {}).get('constraints', {}).get('pass_rate')}")
    return results

if __name__ == "__main__":
    results = run_tests()
    passed = sum(1 for r in results if r['archetype_match'])
    print(f"\nPassed: {passed}/{len(results)} archetype tests")
