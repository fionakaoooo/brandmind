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

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── Tool: archetype_classifier ──────────────────────────────────────────────

def archetype_classifier(brand_brief: str, clip_context: str = "") -> dict:
    """
    Classifies a brand brief into one of 10 archetypes.
    Returns archetype name + rationale.
    """
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

    # validate archetype is one of the 10
    if result["archetype"] not in ARCHETYPES:
        # find closest match by case-insensitive comparison
        for a in ARCHETYPES:
            if a.lower() in result["archetype"].lower():
                result["archetype"] = a
                break
        else:
            result["archetype"] = "Minimal"  # safe fallback

    return result


# ── Constraint extractor ─────────────────────────────────────────────────────

def extract_constraints(brand_brief: str, archetype: str) -> list:
    """
    Pulls explicit and implicit design constraints from the brand brief.
    Returns a list of constraint strings.
    """
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


# ── Planner Agent Node (LangGraph node function) ──────────────────────────────

def planner_agent(state: BrandMindState) -> BrandMindState:
    """
    LangGraph node for the Planner Agent.
    Reads: brand_brief, clip_features
    Writes: archetype, archetype_rationale, design_constraints, status
    """
    print("\n[Planner] Starting brand archetype classification...")

    brand_brief = state["brand_brief"]

    # convert CLIP features to a text summary if present
    clip_context = ""
    if state.get("clip_features"):
        # in real impl, we'd do nearest-neighbor lookup against archetype embeddings
        # for now, just flag that visual context was provided
        clip_context = "Visual brand assets were uploaded (logo/mood board). Incorporate visual tone."

    # step 1: classify archetype
    archetype_result = archetype_classifier(brand_brief, clip_context)
    archetype = archetype_result["archetype"]
    rationale = archetype_result["rationale"]
    print(f"[Planner] Archetype: {archetype}")
    print(f"[Planner] Rationale: {rationale}")

    # step 2: extract design constraints
    constraints = extract_constraints(brand_brief, archetype)
    print(f"[Planner] Extracted {len(constraints)} constraints:")
    for c in constraints:
        print(f"  • {c}")

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


# ── Quick test ────────────────────────────────────────────────────────────────

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
        "iteration_count": 0,
        "status": "planning",
        "revision_history": [],
        "approved_brand_kit": None,
    }

    output_state = planner_agent(initial_state)

    print("\n── Final Planner Output ──")
    print(f"Archetype:    {output_state['archetype']}")
    print(f"Rationale:    {output_state['archetype_rationale']}")
    print(f"Constraints:  {output_state['design_constraints']}")
    print(f"Status:       {output_state['status']}")
