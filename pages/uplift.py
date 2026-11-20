"""Uplift targeting: an ML model that predicts WHO to show the change to."""

import plotly.graph_objects as go
import streamlit as st

import ui
from engine.uplift import targeting_summary, user_types

key = ui.current()
res = ui.get_results(key)

ui.header(
    "🎯 Uplift targeting",
    "A machine-learning model predicts, for each user, how much the change will help them, so we can show it only where it works.",
    """
A normal model predicts *"will this person buy?"*. An **uplift model** predicts *"how much MORE
likely are they to buy **because** we showed them the change?"*. That splits people into four kinds:

| | Buys if NOT shown | Doesn't buy if NOT shown |
|---|---|---|
| **Buys if shown** | 😎 *Sure things*: wasted effort | 🎯 **Persuadables**: target these! |
| **Doesn't buy if shown** | 🐕 *Sleeping dogs*: the change hurts | 🪨 *Lost causes*: wasted effort |

**How we know the model works (the Qini curve):** rank users from "most helped" to "least helped",
then count extra purchases as we treat more of them. A useful model **rises faster than the
straight line** you'd get by picking users at random. Every user is scored by a model that
**never saw them in training**, so this is an honest test.
""",
)

if not res["trust"]["passed"]:
    ui.blocked_banner(res["trust"])
    st.warning("The data is biased, so the model learned from biased data. Shown for illustration only.")

fit = ui.get_uplift(key)
k = fit["per_100k"]
curve_t = fit["curve_t"].copy()
curve_s = fit["curve_s"].copy()
for c in (curve_t, curve_s):
    c[["model", "random"]] *= k

# --- Slider + headline numbers ---------------------------------------------------
top = st.slider("Target the top X% of users (ranked by predicted uplift)", 5, 100, 30, 5,
                format="%d%%") / 100
s = targeting_summary(curve_t, top)
c1, c2, c3 = st.columns(3)
c1.metric(f"Extra purchases: treat top {top:.0%}", f"{s['incremental_top']:+,.0f}",
          help="Per 100,000 visitors, versus showing nobody the change.", border=True)
c2.metric("Extra purchases: treat everyone", f"{s['incremental_everyone']:+,.0f}", border=True)
c3.metric(f"Extra purchases: random {top:.0%}", f"{s['incremental_random']:+,.0f}", border=True)
st.caption("All numbers are extra purchases per 100,000 visitors, measured on users the model was not trained on.")

if s["incremental_everyone"] > 0 and s["incremental_top"] > 0:
    ratio = s["incremental_top"] / s["incremental_everyone"]
    if ratio >= 1:
        st.success(f"Treating only the top **{top:.0%}** gives **{ratio:.1f}×** the extra purchases of treating everyone. "
                   "The rest of users are flat or hurt by the change.")
    else:
        st.info(f"Treating the top **{top:.0%}** captures **{ratio:.0%}** of the extra purchases you'd get from treating everyone.")
st.markdown(f"**Best share to treat:** the curve peaks at **{s['best_fraction']:.0%}** of users "
            f"({s['best_incremental']:+,.0f} extra purchases per 100k visitors)."
            + (" Treating everyone is best, so no targeting is needed." if s["best_fraction"] >= 0.95 else ""))

# --- Qini curve -------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Qini curve: model vs random")
    fig = go.Figure()
    fig.add_scatter(x=curve_t["fraction"], y=curve_t["model"], name="T-learner (two models)",
                    line=dict(color=ui.TREATMENT, width=2.5),
                    hovertemplate="Treat top %{x:.0%}: %{y:+,.0f} extra purchases<extra>T-learner</extra>")
    fig.add_scatter(x=curve_s["fraction"], y=curve_s["model"], name="S-learner (one model)",
                    line=dict(color="#7C3AED", width=2, dash="dot"),
                    hovertemplate="Treat top %{x:.0%}: %{y:+,.0f}<extra>S-learner</extra>")
    fig.add_scatter(x=curve_t["fraction"], y=curve_t["random"], name="Random targeting",
                    line=dict(color=ui.CONTROL, width=2, dash="dash"),
                    hovertemplate="Random %{x:.0%}: %{y:+,.0f}<extra>Random</extra>")
    fig.add_vline(x=top, line_color="#111827", line_width=1, annotation_text=f"top {top:.0%}")
    fig.update_xaxes(tickformat=".0%", title="Share of users treated (highest predicted uplift first)")
    fig.update_yaxes(title="Extra purchases per 100k visitors")
    st.plotly_chart(ui.plotly_layout(fig, height=400))
    st.caption(f"Qini coefficient (area above random; higher is better): T-learner **{fit['qini_t'] * k:,.0f}**, "
               f"S-learner **{fit['qini_s'] * k:,.0f}**. Positive = better than random.")

with right:
    st.subheader("Who is in the audience?")
    types = user_types(fit["scored"]).value_counts(normalize=True)
    order = ["Persuadables", "Sure things", "Lost causes", "Sleeping dogs"]
    colors = [ui.TREATMENT, "#93A3B8", "#CBD5E1", ui.BAD]
    vals = [types.get(t, 0) for t in order]
    fig = go.Figure(go.Bar(x=vals, y=order, orientation="h", marker_color=colors, width=0.55,
                           text=[f"{v:.0%}" for v in vals], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%} of users<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(vals) * 1.25])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(ui.plotly_layout(fig, height=300, showlegend=False))
    st.caption("Persuadables: predicted uplift > +0.2 points. Sleeping dogs: < -0.2 points. "
               "The rest are split by their predicted purchase rate without the change.")

with st.expander("Which users does the model rank highest?"):
    sc = fit["scored"].copy()
    sc["segment"] = (sc["city_tier"].map(lambda t: f"Tier {t}") + " · "
                     + sc["price_band_num"].map({0: "Low", 1: "Mid", 2: "High"}) + " price · "
                     + sc["is_new_user"].map({1: "New", 0: "Returning"}))
    g = sc.groupby("segment").agg(users=("uplift", "size"), predicted_uplift=("uplift", "mean")).reset_index()
    g = g.sort_values("predicted_uplift", ascending=False)
    st.dataframe(g.style.format({"users": "{:,}", "predicted_uplift": "{:+.2%}"}), hide_index=True)
    st.caption("Predicted uplift = predicted change in purchase probability (percentage points).")
