"""Plan a test: sample-size / power calculator."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from engine.stats import sample_size_per_arm

ui.header(
    "🧮 Plan a test",
    "Before launching, work out how many users you need to reliably detect the change you care about.",
    """
Think of trying to hear a whisper in a noisy room: the quieter the whisper (a small effect) and
the noisier the room (a rare event like a purchase), the longer you must listen (more users).

- **Baseline rate**: today's conversion rate.
- **Minimum detectable effect (MDE)**: the smallest lift worth finding, e.g. +10%.
- **Significance (alpha)**: the false-alarm rate we accept (5% = 1 in 20).
- **Power**: the chance of catching the effect *if it's real* (80% is standard). With 80% power,
  1 in 5 real wins would still be missed.

Stopping early when the numbers "look good" breaks these guarantees, so decide the sample size first.
""",
)

c1, c2, c3 = st.columns(3)
baseline = c1.number_input("Baseline conversion rate (%)", 0.1, 50.0, 1.2, 0.1) / 100
mde = c2.number_input("Minimum detectable effect (relative %)", 1.0, 100.0, 10.0, 1.0) / 100
traffic = c3.number_input("Eligible visitors per day (both groups)", 1000, 1_000_000, 8_500, 500)
c4, c5 = st.columns(2)
alpha = c4.select_slider("Significance level (alpha)", [0.01, 0.05, 0.10], value=0.05)
power = c5.select_slider("Power", [0.7, 0.8, 0.9, 0.95], value=0.8)

n = sample_size_per_arm(baseline, mde, alpha, power)
days = math.ceil(2 * n / traffic)
m1, m2, m3 = st.columns(3)
m1.metric("Users per group", f"{n:,}", border=True)
m2.metric("Total users", f"{2 * n:,}", border=True)
m3.metric("Days needed", f"{days:,}", f"≈ {days / 7:.1f} weeks", delta_color="off", delta_arrow="off", border=True)
if days < 7:
    st.caption("Tip: run at least one full week anyway, so every weekday is represented.")

st.subheader("Why rare purchases need leading indicators")
st.markdown(
    "Jewellery is a considered, expensive purchase: only about **1.2%** of visitors buy, while about **8%** add to cart. "
    "Because purchases are so rare, detecting the same relative lift on purchases needs far more users. "
    "That's why teams also track **leading indicators** like add-to-cart: they react faster and confirm direction early, "
    "while the final decision still rests on purchases and guardrails.")

mdes = np.arange(0.05, 0.31, 0.01)
fig = go.Figure()
for label, base, color in [("Purchase (1.2%)", 0.012, ui.TREATMENT), ("Add to cart (8%)", 0.08, ui.CONTROL)]:
    d = [math.ceil(2 * sample_size_per_arm(base, m, alpha, power) / traffic) for m in mdes]
    fig.add_scatter(x=mdes, y=d, name=label, mode="lines", line=dict(color=color, width=2.5),
                    hovertemplate="MDE %{x:+.0%}: %{y:,} days<extra>" + label + "</extra>")
fig.update_xaxes(tickformat="+.0%", title="Minimum detectable effect (relative)")
fig.update_yaxes(type="log", title=f"Days needed at {traffic:,} visitors/day (log scale)")
st.plotly_chart(ui.plotly_layout(fig, height=380))

with st.expander("Reality check: our Price Breakup test"):
    from scipy.stats import norm
    n15 = sample_size_per_arm(0.012, 0.15, 0.05, 0.8)
    # Power of a 60,000-per-group test for a true +15% lift (normal approximation).
    p1, p2 = 0.012, 0.012 * 1.15
    se = math.sqrt(p1 * (1 - p1) / 60_000 + p2 * (1 - p2) / 60_000)
    pw = norm.cdf((p2 - p1) / se - norm.ppf(0.975))
    st.markdown(
        f"To detect +15% on a 1.2% purchase rate with 80% power you need **{n15:,} users per group**. "
        f"Our test had about 60,000 per group, which gives **{pw:.0%} power**: even with a true +15% effect, "
        f"about **{1 - pw:.0%}** of tests this size would come out non-significant. That's why underpowered "
        "'no effect' results should be read as *'we couldn't tell'*, not *'it doesn't work'*.")
