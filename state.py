from typing import TypedDict, Optional


class BrandMindState(TypedDict):
    # --- input ---
    brand_brief: str
    clip_features: Optional[list]       # CLIP embedding from uploaded image (optional)

    # --- planner output ---
    archetype: Optional[str]            # one of the 10 brand archetypes
    archetype_rationale: Optional[str]  # why this archetype fits the brief
    design_constraints: Optional[list]  # extracted constraints from brief

     # --- generator internal ---
    design_spec: Optional[dict]         # structured design intent
    generator_trace: Optional[dict]     # retrieved candidates / reasoning trace

    # --- generator output ---
    draft_brand_kit: Optional[dict]

    # --- qc output ---
    qc_feedback: Optional[str]          # revision instructions if kit fails QC
    qc_scores: Optional[dict]           # wcag ratio, coherence score, etc.

    # --- self-improving heuristic weights ---
    # Maps rule_id -> float weight (default 1.0 for all rules).
    # Initialised by initialise_weights() in planner_agent.
    # Updated by update_heuristic_weights() in qc_agent after every iteration.
    heuristic_weights: Optional[dict]
    
    # --- pipeline control ---
    iteration_count: int                # current revision loop count (max 3)
    status: Optional[str]               # "planning" | "generating" | "reviewing" | "approved" | "failed"
    revision_history: Optional[list]    # log of each draft + QC result
    approved_brand_kit: Optional[dict]  # final output sent to frontend


ARCHETYPES = [
    "Luxury",
    "Tech",
    "Playful",
    "Corporate",
    "Minimal",
    "Bold",
    "Organic",
    "Artisan",
    "Heritage",
    "Youthful",
]
