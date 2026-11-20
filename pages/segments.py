"""Segments: who does the change help or hurt?"""

import plotly.graph_objects as go
import streamlit as st

import ui
from engine.segments import MANY_SEGMENTS
from engine.stats import fmt_p

key = ui.current()
res = ui.get_results(key)
seg = res["segments"]
overall = res["primary"]["rel_lift"]

ui.header(
    "👥 Segments",
    "The average can hide groups that love the change and groups that hate it. Here we split the result by user type.",
    """
Imagine a new spicy dish gets an average rating of 3/5. That might mean everyone thinks it's
"okay"... or that half the diners love it (5/5) and half hate it (1/5). Very different decisions!

Each dot shows the **lift for one group of users**, with a line for its 95% range. **Red** groups move
clearly in the **opposite direction** to the overall result: worth a closer look.

⚠️ **Slicing the data many ways creates false alarms.** Test 20 groups and about one will look
special by pure luck. So we apply a correction (Benjamini-Hochberg) and treat segment results as
**hypotheses to confirm in a follow-up test**, not conclusions.
""",
)

if not res["trust"]["passed"]:
    ui.blocked_banner(res["trust"])
    st.write("")

if len(seg) > MANY_SEGMENTS:
    st.info(f"🔬 **These are hypotheses, not conclusions.** We tested {len(seg)} segments. "
            "Only segments marked ★ survive the multiple-testing correction (adjusted p < 0.05).")

d = seg.sort_values("rel_lift")
fig = go.Figure()
for _, s in d.iterrows():
    color = ui.BAD if (s["against_average"] and s["p_value"] < 0.05) else (ui.TREATMENT if s["significant_adj"] else "#93A3B8")
    star = " ★" if s["significant_adj"] else ""
    fig.add_trace(go.Scatter(
        x=[s["rel_lift"]], y=[s["label"] + star], mode="markers", marker=dict(size=11, color=color),
        error_x=dict(type="data", symmetric=False, array=[s["ci_high"] - s["rel_lift"]],
                     arrayminus=[s["rel_lift"] - s["ci_low"]], thickness=2, width=5, color=color),
        showlegend=False,
        hovertemplate=(f"<b>{s['label']}</b><br>{s['users']:,} users ({s['share_of_users']:.0%})"
                       f"<br>Purchase rate {s['control']:.2%} → {s['treatment']:.2%}"
                       f"<br>Lift {s['rel_lift']:+.1%} (95% CI {s['ci_low']:+.1%} to {s['ci_high']:+.1%})"
                       f"<br>{fmt_p(s['p_value'])}, adjusted p = {s['p_adj']:.3f}<extra></extra>")))
fig.add_vline(x=0, line_dash="dash", line_color="#6B7280")
fig.add_vline(x=overall, line_color=ui.TREATMENT, line_width=1,
              annotation_text=f"overall {overall:+.1%}", annotation_position="top")
fig.update_xaxes(tickformat="+.0%", title="Lift in purchase conversion (95% CI)")
st.plotly_chart(ui.plotly_layout(fig, height=520))
st.caption("Blue ★ = significant after correction · Red = significantly moves against the overall result (p < 0.05) · "
           "Light = could be noise. Hover any dot for details.")

against = seg[seg["against_average"] & (seg["p_value"] < 0.05)]
if len(against):
    st.subheader("Segments moving against the average")
    for _, s in against.iterrows():
        st.markdown(f"- **{s['label']}**: {s['rel_lift']:+.1%} while overall is {overall:+.1%} "
                    f"({fmt_p(s['p_value'])}, adjusted p = {s['p_adj']:.3f}).")

with st.expander("Full segment table"):
    t = seg[["label", "users", "control", "treatment", "rel_lift", "ci_low", "ci_high", "p_value", "p_adj"]].copy()
    st.dataframe(t.style.format({"users": "{:,}", "control": "{:.2%}", "treatment": "{:.2%}", "rel_lift": "{:+.1%}",
                                 "ci_low": "{:+.1%}", "ci_high": "{:+.1%}", "p_value": "{:.4f}", "p_adj": "{:.4f}"}),
                 hide_index=True)
