"""Home: what the tool does, the pipeline, and the recommendation."""

import streamlit as st

import ui
from engine.data import EXPERIMENTS

key = ui.current()
res = ui.get_results(key)
meta = EXPERIMENTS[key]

ui.header(
    "💎 A/B Test Decision Engine",
    "Turns raw experiment data into a trustworthy ship / don't-ship recommendation, "
    "and explains it in plain English.",
    """
An **A/B test** is like a fair coin toss for every visitor: heads, they see the old page
(**control**, grey); tails, they see the new idea (**treatment**, blue). Because the coin decides,
the two groups are alike in every way except the change, so any difference in what they do
is caused by the change (or by luck, which statistics measures).

This tool answers the questions a product manager actually asks:
**Can I trust this test? Did it work? For whom? Did it break anything? What should I do next?**
""",
)

st.markdown(
    f"""
Product teams run dozens of experiments, and reading them well is hard: tests break silently,
small lifts hide inside noise, averages hide winners and losers, and a "win" can quietly hurt
order value or returns. This engine runs every experiment through the same **8-step pipeline**:
trust checks first, then the statistics, guardrails, segments and an ML uplift model. It ends in
**one clear recommendation with reasons**. An AI analyst writes the readout, but only from numbers
the engine computed, and every number it writes is checked.

**Now showing:** {meta['name']}: *{meta['change']}*.
"""
)

ui.decision_card(res["decision"]["decision"], res["decision"]["reasons"])

st.subheader("How the engine works")
steps = [
    ("1", "Data", "Users, variant, funnel events"),
    ("2", "Trust checks", "Is the split fair? (SRM)"),
    ("3", "Primary metric", "Lift, 95% CI, p-value"),
    ("4", "Guardrails", "Did we break AOV, returns, speed?"),
    ("5", "Funnel", "Where in the journey it helps"),
    ("6", "Segments", "Who it helps or hurts"),
    ("7", "Uplift model", "Whom to target (ML)"),
    ("8", "Decision + AI", "Recommendation and readout"),
]
boxes = "".join(
    f"""<div style="border:1px solid #E5E7EB;border-radius:10px;padding:10px 12px;
    background:{'#EFF6FF' if i in (1, 7) else 'white'}">
    <div style="color:{ui.TREATMENT};font-weight:700">{n}. {t}</div>
    <div style="font-size:0.85rem;color:#4B5563">{d}</div></div>"""
    for i, (n, t, d) in enumerate(steps))
# CSS grid: 4 boxes per row on wide screens, fewer on narrow ones.
st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px'>{boxes}</div>",
            unsafe_allow_html=True)
st.caption("Step 2 can stop the pipeline: a broken test gets no recommendation. Step 8 combines explicit rules "
           "with an AI-written explanation.")

st.subheader("The four example experiments")
cols = st.columns(4)
for col, (k, m) in zip(cols, EXPERIMENTS.items()):
    d = ui.get_results(k)["decision"]["decision"]
    color, icon = ui.DECISION_STYLE[d]
    col.markdown(
        f"""<div style="border:1px solid #E5E7EB;border-radius:10px;padding:12px;height:100%">
        <div style="font-weight:700">{m['name']}</div>
        <div style="font-size:0.85rem;color:#4B5563;margin:4px 0 8px">{m['change']}</div>
        <div style="color:{color};font-weight:600">{icon} {d}</div></div>""",
        unsafe_allow_html=True)
st.caption("Pick an experiment in the sidebar; every page updates.")

# Warm-up: train this experiment's uplift model now (~5 s, cached), so the
# Uplift and AI Analyst pages open instantly during a demo.
ui.get_uplift(key)
