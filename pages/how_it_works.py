"""How it works: plain-language walkthrough of the pipeline and tech stack."""

import streamlit as st

import ui

ui.header(
    "📖 How it works",
    "Each step of the engine, what it does, and why it's there, in plain language.",
    """
The guiding principle: **the computer does the maths, explicit rules make the call, and the AI only
explains.** Every step is ordinary, well-understood statistics, written in a few hundred lines of
readable Python and checked by automated tests.
""",
)

STEPS = [
    ("1. Data", "Four realistic jewellery-store experiments, ~120,000 users each, generated in the app from a fixed "
     "random seed. Each has a **known, planted effect**, so we can check the engine finds the truth. "
     "(Price Breakup was planted at +15% and the engine measures +15.0%.)"),
    ("2. Trust checks", "**Sample ratio mismatch** (chi-square test on the 50/50 split), users in both groups, and "
     "balance on pre-experiment traits. If the test is broken, the engine refuses to give a recommendation."),
    ("3. Primary metric", "Purchase conversion compared with a **two-proportion z-test**: lift, 95% confidence "
     "interval and p-value. Verified against the `statsmodels` library, and an A/A simulation (500 fake tests "
     "with no real difference) confirms only ~5% come out 'significant'."),
    ("4. Secondary metrics & guardrails", "View rate, add-to-cart and revenue per user (with **Benjamini-Hochberg** "
     "correction because we test several). Guardrails (order value, returns, page load) use **Welch's t-test**, a "
     "**bootstrap** for skewed order values, and a **non-inferiority** rule: AOV may not drop >3%, returns may not "
     "rise >5%, page load may not rise >5%."),
    ("5. Funnel", "View → add to cart → purchase, per group, showing *where* the change acts."),
    ("6. Segments", "Lift by new/returning, device, city tier, price band and visit history, with corrections and "
     "an explicit 'hypotheses, not conclusions' warning."),
    ("7. Uplift model", "Machine learning (**gradient boosting**, T-learner and S-learner) predicts each user's "
     "personal benefit. Judged on unseen users with a **Qini curve** vs random targeting."),
    ("8. Anomalies, decision & AI readout", "Rules flag broken splits, guardrail breaches, segments moving against "
     "the average, **novelty effects** (lift fading over the two weeks) and daily spikes. Explicit rules turn "
     "everything into **Ship / Targeted rollout / Iterate / Don't ship / Fix the test**. Claude then explains the "
     "result from a JSON fact sheet, and every number it writes is verified."),
]
for title, text in STEPS:
    with st.container(border=True):
        st.markdown(f"**{title}**  \n{text}")

st.subheader("The decision rules")
st.markdown("""
1. Trust checks fail → **No decision (fix the test)**
2. Purchases significantly up → if any guardrail **fails** → **Iterate**; if a segment is significantly hurt → **Targeted rollout**; otherwise → **Ship to all**
3. Purchases significantly down → **Don't ship**
4. Flat overall → if some segment is significantly up → **Targeted rollout**; otherwise → **Don't ship**
""")

st.subheader("Tech stack")
st.markdown("""
| Layer | Tools |
|---|---|
| App & charts | Streamlit (multi-page), Plotly |
| Data | pandas, NumPy (seeded synthetic data) |
| Statistics | SciPy, statsmodels (for verification) |
| Machine learning | scikit-learn `GradientBoostingClassifier` |
| AI analyst | Anthropic Claude API, with a template fallback |
| Quality | pytest: 19 tests, including A/A simulation and planted-effect recovery |
| Hosting | Streamlit Community Cloud, deployed from GitHub |
""")

st.subheader("Limitations")
st.markdown("""
- Data is **synthetic**: realistic in shape, but the effects were designed to demonstrate each method.
- Fixed-horizon tests only (no sequential testing / peeking correction, no CUPED variance reduction yet).
- Segment and uplift findings are hypotheses that should be confirmed with a follow-up test.
- The AI readout is checked for numbers, not for reasoning; the decision itself always comes from the rules.
""")
