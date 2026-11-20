"""AI Analyst: an LLM explains the pre-computed results; we verify its numbers."""

import json

import streamlit as st

import ui
from engine.analyst import MODEL, generate_readout

key = ui.current()
res = ui.get_results(key)

ui.header(
    "🤖 AI Product Analyst",
    "An AI writes the experiment readout in plain English, using only numbers our engine computed, and we check every number it writes.",
    """
Large language models are great writers but can confidently invent numbers ("hallucinate").
So the AI here is **only allowed to explain, never to calculate**:

1. The statistics engine computes everything and packs it into a JSON "fact sheet"
   (you can see it at the bottom of this page).
2. The AI gets the fact sheet plus strict instructions: use only these numbers, follow a fixed
   template, end with three testable hypotheses.
3. **We check its homework:** every number in the AI's text is matched against the fact sheet
   (allowing for rounding). You'll see ✅ if all match, or ⚠️ with a list of numbers that don't.

No API key or no internet? A template writes the same readout from the same numbers, clearly labelled.
""",
)

api_key = ui.get_api_key()
st.caption(f"AI model: `{MODEL}` · " + ("API key found ✅" if api_key else
                                         "no API key configured: the template readout will be used"))

state_key = f"readout_{key}"
if st.button("✨ Generate readout", type="primary"):
    with st.spinner("Writing the readout..."):
        st.session_state[state_key] = generate_readout(ui.get_facts(key), api_key)

out = st.session_state.get(state_key)
if out:
    if out["source"] == "claude":
        st.markdown(f"**Written by Claude** (`{MODEL}`) from the fact sheet below.")
    else:
        st.markdown("**Template readout** (deterministic, built from the same numbers; the AI was not used).")
        if out["note"]:
            st.caption(out["note"])

    v = out["verification"]
    if v["all_verified"]:
        st.success(f"✅ All numbers verified: {v['checked']} numbers checked, every one appears in the fact sheet.")
    else:
        st.warning(f"⚠️ {len(v['unverified'])} number(s) not found in the fact sheet: "
                   + ", ".join(f"`{n}`" for n in v["unverified"]) + ". Treat those statements with caution.")

    with st.container(border=True):
        st.markdown(out["text"])
else:
    st.info("Click **Generate readout** to have the analyst write up this experiment.")

st.subheader("Anomalies found by the rule engine")
if res["anomalies"]:
    icon = {"high": "🔴", "medium": "🟠", "info": "🔵"}
    for a in res["anomalies"]:
        st.markdown(f"{icon[a['severity']]} **{a['title']}**: {a['detail']}")
else:
    st.markdown("🟢 No anomalies detected.")
st.caption("Rules: sample ratio mismatch, guardrail breaches, segments moving against the average, "
           "effects fading over time (novelty), and daily spikes vs the recent average.")

with st.expander("🔍 Show the exact JSON the AI saw"):
    st.code(json.dumps(ui.get_facts(key), indent=2, ensure_ascii=False), language="json")
