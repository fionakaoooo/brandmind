"""
BrandMind Streamlit Frontend
Milestone 3

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="BrandMind",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0D1117; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

.bm-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid #21262D;
}
.bm-logo-mark {
    width: 34px; height: 34px; background: #58A6FF; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #0D1117; font-weight: 700;
}
.bm-brand-name { font-size: 20px; font-weight: 600; color: #F0F6FC; letter-spacing: -0.02em; }
.bm-subtitle { font-size: 13px; color: #8B949E; margin-left: auto; }

.section-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: #8B949E; margin-bottom: 8px; font-weight: 500;
}

.archetype-block {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem;
}
.archetype-name {
    font-size: 22px; font-weight: 600; color: #58A6FF;
    letter-spacing: -0.02em; margin-bottom: 4px;
}
.archetype-rationale { font-size: 13px; color: #8B949E; line-height: 1.6; }

.score-row { display: flex; gap: 10px; margin-bottom: 1rem; }
.score-card {
    flex: 1; background: #161B22; border: 1px solid #21262D;
    border-radius: 10px; padding: 12px; text-align: center;
}
.score-val { font-size: 22px; font-weight: 600; }
.score-val.green { color: #3FB950; }
.score-val.yellow { color: #D29922; }
.score-val.red { color: #F85149; }
.score-label {
    font-size: 11px; color: #8B949E; margin-top: 3px;
    text-transform: uppercase; letter-spacing: 0.06em;
}

.kit-card {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem;
}

.font-specimen { margin-bottom: 10px; }
.font-display { font-size: 18px; font-weight: 500; color: #F0F6FC; }
.font-meta { font-size: 12px; color: #8B949E; margin-top: 2px; }

.palette-row { display: flex; gap: 8px; }
.swatch-col { flex: 1; }
.swatch { height: 44px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); }
.swatch-hex {
    font-size: 10px; color: #8B949E; text-align: center;
    margin-top: 3px; font-family: monospace;
}

.tone-chip {
    display: inline-block; background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.2); color: #79B8FF;
    border-radius: 20px; padding: 3px 10px; font-size: 12px;
    margin: 3px 3px 3px 0;
}

.constraint-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 13px; padding: 7px 0;
    border-bottom: 1px solid #21262D; color: #C9D1D9;
}
.constraint-row:last-child { border-bottom: none; }
.badge-pass  { color: #3FB950; font-size: 12px; }
.badge-fail  { color: #F85149; font-size: 12px; }
.badge-uncertain { color: #D29922; font-size: 12px; }

.rule-item {
    font-size: 13px; color: #8B949E; line-height: 1.55;
    padding: 5px 0; border-bottom: 1px solid #1C2129;
    display: flex; gap: 8px;
}
.rule-item:last-child { border-bottom: none; }

.alignment-block {
    background: #0D1117; border: 1px solid #21262D;
    border-left: 3px solid #58A6FF;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 13px; color: #8B949E; line-height: 1.7;
    margin-top: 6px; font-style: italic;
}

.iteration-timeline {
    display: flex; gap: 6px; margin-bottom: 1rem;
}
.iter-dot {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; border: 1px solid #30363D;
}
.iter-dot.pass  { background: rgba(63,185,80,0.15);  color: #3FB950; border-color: #3FB950; }
.iter-dot.fail  { background: rgba(248,81,73,0.15);  color: #F85149; border-color: #F85149; }
.iter-dot.best  { background: rgba(88,166,255,0.15); color: #58A6FF; border-color: #58A6FF; }
.iter-label { font-size: 11px; color: #8B949E; margin-top: 2px; text-align: center; }

.status-tag {
    display: inline-block; background: #21262D; border-radius: 6px;
    padding: 3px 8px; font-size: 11px; color: #8B949E; margin-bottom: 1rem;
}
.status-tag.approved { background: rgba(63,185,80,0.15); color: #3FB950; }
.status-tag.failed   { background: rgba(248,81,73,0.10); color: #F85149; }

.empty-state {
    border: 1px dashed #21262D; border-radius: 12px;
    padding: 4rem 2rem; text-align: center;
}
.empty-title { font-size: 14px; font-weight: 500; color: #8B949E; margin-bottom: 6px; }
.empty-body  { font-size: 13px; color: #484F58; line-height: 1.7; }

.stTextArea textarea {
    background: #161B22 !important; border: 1px solid #30363D !important;
    border-radius: 10px !important; color: #F0F6FC !important;
    font-size: 14px !important; line-height: 1.6 !important;
}
.stTextArea textarea:focus { border-color: #58A6FF !important; }
.stButton > button {
    background: #58A6FF !important; color: #0D1117 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 14px !important;
    width: 100% !important; padding: 0.65rem 1rem !important;
}
.stButton > button:hover { background: #79B8FF !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="bm-header">
    <div class="bm-logo-mark">B</div>
    <span class="bm-brand-name">BrandMind</span>
    <span class="bm-subtitle">Agentic brand identity generation · 3-agent pipeline</span>
</div>
""", unsafe_allow_html=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def cclass(v, hi=0.8, lo=0.5):
    return "green" if v >= hi else ("yellow" if v >= lo else "red")

def badge(s):
    if s == "pass":      return '<span class="badge-pass">✓ Pass</span>'
    if s == "fail":      return '<span class="badge-fail">✗ Fail</span>'
    return '<span class="badge-uncertain">? Uncertain</span>'


# ── Layout ────────────────────────────────────────────────────────────────────

col_input, col_output = st.columns([1, 1.3], gap="large")

with col_input:
    st.markdown('<div class="section-label">Brand brief</div>', unsafe_allow_html=True)

    brief = st.text_area(
        label="brief",
        label_visibility="collapsed",
        value=(
            "We are launching a sustainable skincare brand called Verdant "
            "targeting eco-conscious women aged 25–40. The brand should feel "
            "premium but warm and grounded in nature. The color palette must be "
            "WCAG AA accessible. No neon colors."
        ),
        height=200,
        placeholder=(
            "Describe your brand — industry, target audience, tone, values, "
            "and any explicit constraints (e.g. WCAG accessible, no serif fonts, no red)..."
        ),
    )

    st.markdown("""
    <div style="margin:0.75rem 0 1rem;font-size:12px;color:#8B949E;line-height:1.9">
        <span style="color:#58A6FF;font-weight:500">Agent 1</span>
        &nbsp;&nbsp;classifies archetype + extracts constraints<br>
        <span style="color:#58A6FF;font-weight:500">Agent 2</span>
        &nbsp;&nbsp;retrieves fonts, palette &amp; design rules<br>
        <span style="color:#58A6FF;font-weight:500">Agent 3</span>
        &nbsp;&nbsp;WCAG check · coherence score · constraint verify
    </div>
    """, unsafe_allow_html=True)

    generate = st.button("✦  Generate brand kit")

    if generate and brief.strip():
        loading_steps = [
            ("Agent 1", "Classifying brand archetype..."),
            ("Agent 1", "Extracting explicit & implicit constraints..."),
            ("Agent 2", "Retrieving font candidates from Google Fonts..."),
            ("Agent 2", "Scoring color palettes against EmoSet targets..."),
            ("Agent 2", "Applying design heuristics..."),
            ("Agent 3", "Running WCAG 2.1 AA compliance check..."),
            ("Agent 3", "Scoring archetype coherence (LLM judge)..."),
            ("Agent 3", "Verifying constraint satisfaction..."),
        ]
        placeholder = st.empty()
        for agent, text in loading_steps:
            placeholder.markdown(
                f"<div style='font-size:13px;color:#8B949E;padding:4px 0'>"
                f"<span style='color:#58A6FF;font-weight:500'>{agent}</span>"
                f"&nbsp;&nbsp;{text}</div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.45)
        placeholder.empty()

        try:
            from graph import run_pipeline
            state = run_pipeline(brand_brief=brief.strip(), max_iterations=3)
            st.session_state["result"] = state
            st.session_state["error"] = None
        except Exception as e:
            st.session_state["error"] = str(e)
            st.session_state["result"] = None


# ── Output panel ──────────────────────────────────────────────────────────────

with col_output:

    if st.session_state.get("error"):
        st.error(f"Pipeline error: {st.session_state['error']}")

    elif st.session_state.get("result"):
        state     = st.session_state["result"]
        kit       = state.get("approved_brand_kit") or state.get("draft_brand_kit") or {}
        qc        = state.get("qc_scores") or {}
        status    = state.get("status", "—")
        iterations = state.get("iteration_count", 0)
        history   = state.get("revision_history") or []

        # ── Status tag ────────────────────────────────────────────────────────
        tag_cls  = "approved" if status == "approved" else "failed"
        tag_text = "approved ✓" if status == "approved" else f"max iterations ({iterations})"
        st.markdown(
            f'<div class="status-tag {tag_cls}">Pipeline: {tag_text}</div>',
            unsafe_allow_html=True,
        )

        # ── Iteration timeline ────────────────────────────────────────────────
        if history:
            best_iter = max(
                range(len(history)),
                key=lambda i: (
                    (history[i].get("qc_scores") or {}).get("constraints", {}).get("pass_count", 0),
                    (history[i].get("qc_scores") or {}).get("overall_score", 0),
                )
            )
            dots = ""
            for i, entry in enumerate(history):
                entry_status = entry.get("status", "failed")
                overall = (entry.get("qc_scores") or {}).get("overall_score", 0)
                dot_cls = "pass" if entry_status == "approved" else ("best" if i == best_iter else "fail")
                dots += (
                    f'<div style="text-align:center">'
                    f'<div class="iter-dot {dot_cls}">{i+1}</div>'
                    f'<div class="iter-label">{overall:.2f}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="margin-bottom:1rem">'
                f'<div class="section-label">Revision history</div>'
                f'<div class="iteration-timeline">{dots}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Archetype ─────────────────────────────────────────────────────────
        archetype = kit.get("archetype") or state.get("archetype") or "—"
        rationale = state.get("archetype_rationale") or "—"
        alignment = kit.get("archetype_alignment", "")

        st.markdown(f"""
        <div class="archetype-block">
            <div class="section-label">Archetype · Agent 1</div>
            <div class="archetype-name">{archetype}</div>
            <div class="archetype-rationale">{rationale}</div>
            {f'<div class="alignment-block">{alignment}</div>' if alignment else ''}
        </div>
        """, unsafe_allow_html=True)

        # ── QC scores ─────────────────────────────────────────────────────────
        wcag_rate = qc.get("wcag", {}).get("pass_rate", 0)
        coherence = qc.get("coherence", {}).get("score", 0)
        con_rate  = qc.get("constraints", {}).get("pass_rate", 0)

        st.markdown(f"""
        <div class="section-label" style="margin-top:0.25rem">QC scores · Agent 3</div>
        <div class="score-row">
            <div class="score-card">
                <div class="score-val {cclass(coherence/5,0.8,0.7)}">{coherence:.1f}<span style="font-size:13px;color:#8B949E">/5</span></div>
                <div class="score-label">Archetype coherence</div>
            </div>
            <div class="score-card">
                <div class="score-val {cclass(wcag_rate)}">{int(wcag_rate*100)}%</div>
                <div class="score-label">WCAG AA pass rate</div>
            </div>
            <div class="score-card">
                <div class="score-val {cclass(con_rate)}">{int(con_rate*100)}%</div>
                <div class="score-label">Constraint satisfaction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Font pairing ──────────────────────────────────────────────────────
        fonts    = kit.get("font_recommendation") or {}
        headline = fonts.get("headline") or {}
        body_f   = fonts.get("body") or {}
        h_name   = headline.get("family", "—")
        b_name   = body_f.get("family", "—")
        h_cat    = headline.get("category", "")
        b_cat    = body_f.get("category", "")
        pairing  = (fonts.get("pairing_rationale") or "")[:140]

        st.markdown(f"""
        <div class="kit-card">
            <div class="section-label">Font pairing · Agent 2</div>
            <div class="font-specimen">
                <div class="font-display">{h_name}</div>
                <div class="font-meta">Headline · {h_cat}</div>
            </div>
            <div class="font-specimen">
                <div class="font-display">{b_name}</div>
                <div class="font-meta">Body · {b_cat}</div>
            </div>
            <div style="font-size:12px;color:#8B949E;margin-top:8px;font-style:italic">{pairing}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Color palette ─────────────────────────────────────────────────────
        palette_data = kit.get("color_palette") or {}
        hex_codes    = [
            h for h in (palette_data.get("hex_codes") or [])
            if isinstance(h, str) and h.startswith("#") and len(h) == 7
        ][:5]

        if hex_codes:
            swatches = "".join([
                f'<div class="swatch-col">'
                f'<div class="swatch" style="background:{h}"></div>'
                f'<div class="swatch-hex">{h}</div>'
                f'</div>'
                for h in hex_codes
            ])
            emotions     = palette_data.get("matched_emotions") or []
            emotion_html = "".join([f'<span class="tone-chip">{e}</span>' for e in emotions[:4]])
            pal_note     = (palette_data.get("palette_rationale") or "")[:120]
            st.markdown(f"""
            <div class="kit-card">
                <div class="section-label">Color palette · Agent 2</div>
                <div class="palette-row">{swatches}</div>
                <div style="margin-top:10px">{emotion_html}</div>
                {f'<div style="font-size:12px;color:#484F58;margin-top:8px;line-height:1.5">{pal_note}</div>' if pal_note else ''}
            </div>
            """, unsafe_allow_html=True)

        # ── Tone & voice ──────────────────────────────────────────────────────
        tone_seed    = kit.get("tone_and_voice_seed") or {}
        tone_kws     = tone_seed.get("tone_keywords") or []
        palette_note = tone_seed.get("palette_notes", "")

        if tone_kws:
            chips = "".join([f'<span class="tone-chip">{t}</span>' for t in tone_kws[:8]])
            st.markdown(f"""
            <div class="kit-card">
                <div class="section-label">Tone &amp; voice · Agent 2</div>
                <div>{chips}</div>
                <div style="font-size:13px;color:#8B949E;margin-top:10px;line-height:1.6">{palette_note}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Design rules ──────────────────────────────────────────────────────
        rules = (kit.get("design_rules") or [])[:5]
        if rules:
            rules_html = "".join([
                f'<div class="rule-item">'
                f'<span style="color:#3FB950;flex-shrink:0">·</span>'
                f'<span>{r}</span></div>'
                for r in rules
            ])
            st.markdown(f"""
            <div class="kit-card">
                <div class="section-label">Design rules · Agent 2</div>
                {rules_html}
            </div>
            """, unsafe_allow_html=True)

        # ── Constraint verification ───────────────────────────────────────────
        con_items = (qc.get("constraints") or {}).get("items") or []
        if con_items:
            rows = "".join([
                f'<div class="constraint-row">'
                f'<span>{item.get("constraint","")[:60]}</span>'
                f'{badge(item.get("status",""))}'
                f'</div>'
                for item in con_items[:8]
            ])
            pass_n  = sum(1 for i in con_items if i.get("status") == "pass")
            total_n = len(con_items)
            st.markdown(f"""
            <div class="kit-card">
                <div class="section-label">Constraint verification · Agent 3 &nbsp;
                    <span style="color:#8B949E;font-weight:400;text-transform:none;letter-spacing:0">
                        {pass_n}/{total_n} passed
                    </span>
                </div>
                {rows}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:28px;margin-bottom:1rem;color:#30363D">✦</div>
            <div class="empty-title">Your brand kit will appear here</div>
            <div class="empty-body">
                Enter a brand brief on the left and click Generate.<br>
                The 3-agent pipeline will classify your archetype,<br>
                retrieve fonts and palette, then run QC automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
