"""Trust checks: is the experiment itself healthy?"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from engine.stats import fmt_p

key = ui.current()
trust = ui.get_results(key)["trust"]
srm, dup = trust["srm"], trust["duplicates"]

ui.header(
    "🛡️ Trust checks",
    "Before reading any result, we check that the experiment ran correctly. A broken test gives confident-looking but wrong answers.",
    """
**Sample ratio mismatch (SRM).** We asked a fair coin to split users 50/50. Flip a fair coin
120,000 times and you'll get *close* to 60,000 heads, maybe off by a few hundred. If you got
61,000 vs 59,000, you'd suspect the coin. Here the "coin" is our code. An uneven split usually means some
users were **lost or duplicated in one group** (a crash, a slow page, a logging bug), and those
lost users are rarely random, so the groups stop being comparable.

The **p-value** answers: *"if the split really were fair, how often would we see a gap this big?"*
We only raise the alarm below 0.001 (1 in 1,000), so a healthy test almost never trips it.

**Duplicates:** a user who saw both versions contaminates both groups.

**Balance:** things decided *before* the test (how many past visits, new vs returning) should
look the same in both groups. If they don't, randomisation went wrong.
""",
)

if not trust["passed"]:
    ui.blocked_banner(trust)
    st.write("")
else:
    st.success("✅ All trust checks passed: the groups are comparable, so the results can be read.")

c1, c2, c3 = st.columns(3)
c1.metric("Control users", f"{srm['n_control']:,}")
c2.metric("Treatment users", f"{srm['n_treatment']:,}",
          f"{srm['n_treatment'] - srm['n_control']:+,} vs control", delta_color="off", delta_arrow="off")
c3.metric("Treatment share", f"{srm['share_treatment']:.2%}", "planned 50.00%", delta_color="off", delta_arrow="off")

left, right = st.columns([3, 2])
with left:
    st.subheader("1. Sample ratio (SRM)")
    status = "✅ Pass" if srm["passed"] else "❌ Fail"
    st.markdown(f"**{status}**: chi-square = {srm['chi2']:.2f}, {fmt_p(srm['p_value'])} "
                f"(alarm if p < 0.001).")
    expected = (srm["n_control"] + srm["n_treatment"]) / 2
    fig = go.Figure()
    for arm, n in [("control", srm["n_control"]), ("treatment", srm["n_treatment"])]:
        fig.add_bar(x=[arm.capitalize()], y=[n], name=arm.capitalize(), marker_color=ui.ARM_COLORS[arm],
                    width=0.45, text=[f"{n:,}"], textposition="outside",
                    hovertemplate=f"{arm.capitalize()}: %{{y:,}} users<extra></extra>")
    fig.add_hline(y=expected, line_dash="dash", line_color="#111827",
                  annotation_text=f"expected {expected:,.0f}", annotation_position="top left")
    lo = min(srm["n_control"], srm["n_treatment"])
    fig.update_yaxes(range=[lo * 0.97, max(srm["n_control"], srm["n_treatment"]) * 1.015], title="Users")
    st.plotly_chart(ui.plotly_layout(fig, height=320, showlegend=False), use_container_width=True)
    st.caption("The y-axis is zoomed in to make the gap visible.")

with right:
    st.subheader("2. Duplicate users")
    st.markdown(f"**{'✅ Pass' if dup['passed'] else '❌ Fail'}**: {dup['users_in_both_arms']} users in both groups, "
                f"{dup['duplicate_rows']} duplicate rows.")
    st.subheader("3. Pre-experiment balance")
    bal = pd.DataFrame(trust["balance"])
    bal["Control"] = [f"{v:.2f}" if "average" in a else f"{v:.1%}" for a, v in zip(bal.attribute, bal.control)]
    bal["Treatment"] = [f"{v:.2f}" if "average" in a else f"{v:.1%}" for a, v in zip(bal.attribute, bal.treatment)]
    bal["p-value"] = bal["p_value"].map(lambda p: f"{p:.3f}")
    bal["Status"] = bal["passed"].map({True: "✅ Balanced", False: "❌ Imbalanced"})
    st.dataframe(bal[["attribute", "Control", "Treatment", "p-value", "Status"]].rename(columns={"attribute": "Attribute"}),
                 hide_index=True, use_container_width=True)
    if not trust["passed"]:
        st.info("Notice the balance checks can still pass when SRM fails: the users lost here (slow page loads) "
                "aren't different in visit history, but they *are* less likely to buy. That's why SRM is checked on its own.")
