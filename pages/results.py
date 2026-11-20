"""Results: primary metric, all metrics with CIs, guardrails, funnel."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from engine.stats import fmt_p

key = ui.current()
res = ui.get_results(key)
p = res["primary"]

ui.header(
    "📊 Results",
    "Did the change move the numbers we care about, by how much, and could it just be luck?",
    """
**Lift** = how much better the new version did, e.g. *+15%* means 1.20% of visitors bought
before and 1.38% after.

**95% confidence interval (CI)** is like a weather forecast range: *"tomorrow 28–32°C"*.
We can't know the exact true effect, but we're 95% confident it lies in this range. **If the
whole range is above zero, the change really helped.** If the range crosses zero, it might
have done nothing.

**p-value** answers *"if the change did nothing at all, how often would luck alone produce a
difference this big?"* Below 0.05 (1 in 20) we call it statistically significant: probably real.

**Guardrails** are metrics we promise not to break while chasing the main goal (order value,
returns, page speed), each with an agreed tolerance.
""",
)

if not res["trust"]["passed"]:
    ui.blocked_banner(res["trust"])
    st.warning("Numbers below are shown for illustration only. Do not act on them.")

# --- Scorecard tiles -------------------------------------------------------------
if p["significant"]:
    verdict = "Real effect" if p["rel_lift"] > 0 else "Real, but negative"
    words = "Very unlikely to be luck" if p["p_value"] < 0.01 else "Unlikely to be luck"
else:
    verdict, words = "Could be luck", "Not distinguishable from zero"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Purchase conversion lift", ui.pct(p["rel_lift"]),
          f"{p['control']:.2%} → {p['treatment']:.2%}", delta_color="off", delta_arrow="off", border=True)
c2.metric("95% confidence interval", f"{ui.pct(p['rel_ci'][0], 0)} … {ui.pct(p['rel_ci'][1], 0)}",
          ("entirely above 0" if p["rel_ci"][0] > 0 else "includes 0")
          + f" ({ui.pct(p['rel_ci'][0])} to {ui.pct(p['rel_ci'][1])})", delta_color="off", delta_arrow="off", border=True)
c3.metric("p-value", fmt_p(p["p_value"]).replace("p = ", "").replace("p ", ""),
          "below 0.05" if p["significant"] else "above 0.05", delta_color="off", delta_arrow="off", border=True)
c4.metric("Real or luck?", verdict, words, delta_color="off", delta_arrow="off", border=True)

# --- Forest plot -----------------------------------------------------------------
st.subheader("Every metric at a glance")
st.caption("Each dot is the relative change; the line is its 95% CI. Lines that cross the dashed zero line "
           "might be no change at all. Guardrail tolerances are marked with ┃.")
rows = [p] + res["secondary"] + res["guardrails"]
labels = [f"{m['name']}  ({'primary' if m['kind'] == 'primary' else m['kind']})" for m in rows][::-1]
rows = rows[::-1]
fig = go.Figure()
for label, m in zip(labels, rows):
    lo, hi = m["rel_ci"]
    if m["kind"] == "guardrail":
        color = {"PASS": ui.GOOD, "WATCH": ui.WARN, "FAIL": ui.BAD}[m["status"]]
    else:
        color = ui.TREATMENT if (m["p_value"] < 0.05) else ui.CONTROL
    fig.add_trace(go.Scatter(
        x=[m["rel_lift"]], y=[label], mode="markers", marker=dict(size=11, color=color),
        error_x=dict(type="data", symmetric=False, array=[hi - m["rel_lift"]], arrayminus=[m["rel_lift"] - lo],
                     thickness=2, width=6, color=color),
        showlegend=False,
        hovertemplate=(f"<b>{m['name']}</b><br>Change: {m['rel_lift']:+.1%}<br>95% CI: {lo:+.1%} to {hi:+.1%}"
                       f"<br>{fmt_p(m['p_value'])}<extra></extra>")))
    if m["kind"] == "guardrail":
        tol = m["tolerance"] if m["bad_direction"] == "increase" else -m["tolerance"]
        fig.add_trace(go.Scatter(x=[tol], y=[label], mode="markers", showlegend=False,
                                 marker=dict(symbol="line-ns-open", size=18, color="#111827", line=dict(width=2)),
                                 hovertemplate=f"Tolerance: {tol:+.0%}<extra></extra>"))
fig.add_vline(x=0, line_dash="dash", line_color="#6B7280")
fig.update_xaxes(tickformat="+.0%", title="Relative change, treatment vs control")
st.plotly_chart(ui.plotly_layout(fig, height=360))

# --- Guardrail table ---------------------------------------------------------
st.subheader("Guardrails")
g = pd.DataFrame([{
    "Guardrail": m["name"],
    "Control": f"₹{m['control']:,.0f}" if m["unit"] == "rupees" else (f"{m['control']:.0f} ms" if m["unit"] == "ms" else f"{m['control']:.1%}"),
    "Treatment": f"₹{m['treatment']:,.0f}" if m["unit"] == "rupees" else (f"{m['treatment']:.0f} ms" if m["unit"] == "ms" else f"{m['treatment']:.1%}"),
    "Change": ui.pct(m["rel_lift"]),
    "95% CI": f"{ui.pct(m['rel_ci'][0])} to {ui.pct(m['rel_ci'][1])}",
    "Tolerance": f"{'rise' if m['bad_direction'] == 'increase' else 'drop'} ≤ {m['tolerance']:.0%}",
    "Status": ui.STATUS_STYLE[m["status"]][1],
} for m in res["guardrails"]])
st.dataframe(g, hide_index=True)
st.caption("✅ Pass: even the worst case in the CI is within tolerance.  ❌ Fail: significantly worse than tolerance.  "
           "👀 Watch: too noisy to prove safe, but no evidence of harm, so monitor after launch.")

# --- Funnel + daily trend ----------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Funnel")
    f = res["funnel"]
    fig = go.Figure()
    for arm in ["control", "treatment"]:
        d = f[f.variant == arm]
        fig.add_bar(x=d["step"], y=d["overall_rate"], name=arm.capitalize(), marker_color=ui.ARM_COLORS[arm],
                    customdata=d[["users", "step_rate"]].to_numpy(),
                    text=[f"{v:.2%}" for v in d["overall_rate"]], textposition="outside",
                    hovertemplate="%{x}<br>%{y:.2%} of all users<br>%{customdata[0]:,} users"
                                  "<br>%{customdata[1]:.1%} of previous step<extra>" + arm + "</extra>")
    fig.update_yaxes(type="log", tickformat=".1%", title="Share of all users (log scale)")
    st.plotly_chart(ui.plotly_layout(fig, height=360, bargap=0.25, bargroupgap=0.08))
    st.caption("Log scale, so the small purchase step stays visible. Hover for step-to-step conversion.")
with right:
    st.subheader("Purchase rate by day")
    d = res["daily"]
    fig = go.Figure()
    for arm in ["control", "treatment"]:
        a = d[d.variant == arm]
        fig.add_scatter(x=a["exposure_day"], y=a["rate"], name=arm.capitalize(), mode="lines+markers",
                        line=dict(color=ui.ARM_COLORS[arm], width=2), marker=dict(size=8),
                        hovertemplate="Day %{x}: %{y:.2%}<extra>" + arm + "</extra>")
    fig.update_xaxes(title="Day of experiment", dtick=1)
    fig.update_yaxes(tickformat=".1%", title="Purchase rate")
    st.plotly_chart(ui.plotly_layout(fig, height=360))
    st.caption("Daily numbers are noisy (~8,500 users a day). Look for patterns, like a gap that shrinks over time.")
