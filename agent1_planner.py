
"""
Agent 1: Planner Agent
- Receives brand brief (+ optional CLIP features)
- Classifies into one of 10 brand archetypes
- Extracts explicit design constraints
- Writes archetype + constraints to shared LangGraph state
"""

import os
import json
from openai import OpenAI
from state import BrandMindState, ARCHETYPES
from tools.heuristic_search import initialise_weights

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── Tool: archetype_classifier ──────────────────────────────────────────────

def archetype_classifier(brand_brief: str, clip_context: str = "") -> dict:
    archetype_list = "\n".join(f"- {a}" for a in ARCHETYPES)

    clip_section = ""
    if clip_context:
        clip_section = f"\nVisual context from uploaded image: {clip_context}\n"

    prompt = f"""You are a senior brand strategist. Given the brand brief below, classify it into 
exactly one of the following 10 brand archetypes. Choose the single best fit.

Archetypes:
{archetype_list}

Brand brief:
{brand_brief}
{clip_section}
Respond ONLY with valid JSON in this exact format:
{{
  "archetype": "<one of the 10 archetypes above>",
  "rationale": "<1-2 sentences explaining why this archetype fits best>"
}}"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    result = json.loads(resp.choices[0].message.content)

    if result["archetype"] not in ARCHETYPES:
        for a in ARCHETYPES:
            if a.lower() in result["archetype"].lower():
                result["archetype"] = a
                break
        else:
            result["archetype"] = "Minimal"

    return result


# ── Constraint extractor ─────────────────────────────────────────────────────

def extract_constraints(brand_brief: str, archetype: str) -> list:
    prompt = f"""You are a brand design expert. Given the brand brief and its archetype, 
extract all design constraints — both explicit (directly stated) and implicit (reasonably implied).

Brand archetype: {archetype}
Brand brief: {brand_brief}

Constraints to look for include:
- Accessibility requirements (e.g., WCAG compliance, colorblind-friendly)
- Audience-driven tone requirements (e.g., "approachable for children" → avoid harsh fonts)
- Cultural or regional considerations
- Industry conventions that must be respected
- Explicit style restrictions (e.g., "no serif fonts", "must feel minimal")
- Color restrictions (e.g., "no red", "must use brand color #2E86AB")

Respond ONLY with valid JSON:
{{
  "constraints": [
    "constraint 1 as a clear, actionable string",
    "constraint 2",
    ...
  ]
}}

If no constraints are found, return {{"constraints": []}}."""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    result = json.loads(resp.choices[0].message.content)
    return result.get("constraints", [])


# ── Planner Agent Node ────────────────────────────────────────────────────────

def planner_agent(state: BrandMindState) -> BrandMindState:
    print("\n[Planner] Starting brand archetype classification...")

    brand_brief = state["brand_brief"]

    clip_context = ""
    if state.get("clip_features"):
        clip_context = "Visual brand assets were uploaded (logo/mood board). Incorporate visual tone."

    # step 1: classify archetype
    archetype_result = archetype_classifier(brand_brief, clip_context)
    archetype = archetype_result["archetype"]
    rationale = archetype_result["rationale"]
    print(f"[Planner] Archetype: {archetype}")
    print(f"[Planner] Rationale: {rationale}")

    # step 2: extract design constraints
    constraints = extract_constraints(brand_brief, archetype)

    # step 3: force-inject literal constraints from brief
    brief_lower = brand_brief.lower()
    forced = []
    if "wcag" in brief_lower or "accessible" in brief_lower:
        forced.append("Color palette must be WCAG AA accessible")
    if "no neon" in brief_lower:
        forced.append("No neon colors")
    if "no warm" in brief_lower or "no earthy" in brief_lower:
        forced.append("No warm or earthy tones")
    if "no serif" in brief_lower:
        forced.append("No serif fonts")
    if "no traditional bank" in brief_lower:
        forced.append("No traditional bank vibes")

    def _norm(s):
        return s.strip().rstrip(".,;").lower()

    seen = set()
    deduped = []
    for c in constraints:
        if _norm(c) not in seen:
            seen.add(_norm(c))
            deduped.append(c)
    constraints = deduped

    for f in forced:
        if _norm(f) not in {_norm(c) for c in constraints}:
            constraints.insert(0, f)

    print(f"[Planner] Extracted {len(constraints)} constraints:")
    for c in constraints:
        print(f"  • {c}")

    state = initialise_weights(state)

    return {
        **state,
        "archetype": archetype,
        "archetype_rationale": rationale,
        "design_constraints": constraints,
        "constraints": constraints,
        "clip_features": clip_context if clip_context else state.get("clip_features"),
        "status": "generating",
        "iteration_count": 0,
        "revision_history": [],
    }


if __name__ == "__main__":
    test_brief = """
    We're launching a sustainable skincare brand called Verdant targeting eco-conscious 
    women aged 25-40. The brand should feel premium but not cold — warm, trustworthy, 
    and grounded in nature. We want to avoid anything that feels synthetic or corporate. 
    The color palette must be WCAG AA accessible. No neon colors. 
    We're aiming for a minimalist but warm aesthetic.
    """

    initial_state: BrandMindState = {
        "brand_brief": test_brief.strip(),
        "clip_features": None,
        "archetype": None,
        "archetype_rationale": None,
        "design_constraints": None,
        "draft_brand_kit": None,
        "qc_feedback": None,
        "qc_scores": None,
        "heuristic_weights": None,
        "iteration_count": 0,
        "status": "planning",
        "revision_history": [],
        "approved_brand_kit": None,
    }

    output_state = planner_agent(initial_state)
    print(f"\nArchetype:   {output_state['archetype']}")
    print(f"Constraints: {output_state['design_constraints']}")
