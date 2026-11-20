"""
Segment analysis: lift on the primary metric within each user group.

Caution built in: when you slice data many ways, some slices will look
special by pure chance (test 20 segments and expect ~1 false "win" at 5%).
We therefore apply Benjamini-Hochberg across all segments and label the
output as hypotheses, not conclusions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, two_proportion_ztest

MANY_SEGMENTS = 5  # above this, show the "hypotheses, not conclusions" note


def _prior_bucket(s: pd.Series) -> pd.Series:
    """Returning users only (new users have 0 sessions and are covered by
    'New vs returning'). New users get NaN, which groupby skips."""
    return pd.cut(s, bins=[0, 3, 9, np.inf], labels=["1-3", "4-9", "10+"])


# (column used for grouping, label shown to users, function to build the column)
DIMENSIONS = [
    ("user_type", "New vs returning", lambda d: np.where(d["is_new_user"] == 1, "New", "Returning")),
    ("device", "Device", lambda d: d["device"].str.capitalize()),
    ("city_tier", "City tier", lambda d: "Tier " + d["city_tier"].astype(str)),
    ("price_band", "Price band", lambda d: d["price_band"].str.capitalize()),
    ("prior_bucket", "Returning, prior sessions", lambda d: _prior_bucket(d["prior_sessions"])),
]


def segment_lifts(df: pd.DataFrame, metric: str = "purchased") -> pd.DataFrame:
    """Return one row per segment with control/treatment rates, lift, CI and
    BH-adjusted p-value. `against_average` marks segments whose lift has the
    opposite sign to the overall lift (a classic thing to investigate)."""
    overall = two_proportion_ztest(
        int(df.loc[df.variant == "control", metric].sum()), int((df.variant == "control").sum()),
        int(df.loc[df.variant == "treatment", metric].sum()), int((df.variant == "treatment").sum()),
    )
    rows = []
    for col, dim_label, build in DIMENSIONS:
        seg = pd.Series(build(df), index=df.index)
        for level, g in df.groupby(seg, observed=True):
            c, t = g[g.variant == "control"], g[g.variant == "treatment"]
            if len(c) == 0 or len(t) == 0 or c[metric].sum() == 0:
                continue
            r = two_proportion_ztest(int(c[metric].sum()), len(c), int(t[metric].sum()), len(t))
            rows.append({
                "dimension": dim_label, "segment": str(level),
                "label": f"{dim_label}: {level}",
                "users": len(g), "share_of_users": len(g) / len(df),
                "control": r["control"], "treatment": r["treatment"],
                "rel_lift": r["rel_lift"], "ci_low": r["rel_ci"][0], "ci_high": r["rel_ci"][1],
                "p_value": r["p_value"],
            })
    out = pd.DataFrame(rows)
    out["p_adj"], out["significant_adj"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["against_average"] = np.sign(out["rel_lift"]) != np.sign(overall["rel_lift"])
    return out.sort_values("rel_lift", ascending=False).reset_index(drop=True)
