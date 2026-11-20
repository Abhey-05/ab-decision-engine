"""
Rule-based anomaly detection. Each rule is simple and explainable; the AI
analyst only *describes* what these rules found.

Each anomaly is a dict: {severity: "high"|"medium"|"info", title, detail}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import fmt_p, two_proportion_ztest

EARLY_DAYS = (1, 4)
LATE_DAYS = (8, 14)
SPIKE_Z = 3.0          # how many standard errors counts as a spike
ROLLING_WINDOW = 5     # previous days used as the "normal" level


def _lift_for_days(df: pd.DataFrame, first: int, last: int) -> dict:
    d = df[df["exposure_day"].between(first, last)]
    c, t = d[d.variant == "control"], d[d.variant == "treatment"]
    return two_proportion_ztest(int(c.purchased.sum()), len(c), int(t.purchased.sum()), len(t))


def novelty_check(df: pd.DataFrame) -> dict:
    """Compare the lift in the first days with the lift in the last week.

    A novelty effect = users react to something because it's new, then get
    used to it. If early lift is significant and clearly larger than late
    lift (difference test p < 0.05), the long-run effect is probably smaller
    than the headline number.
    """
    early = _lift_for_days(df, *EARLY_DAYS)
    late = _lift_for_days(df, *LATE_DAYS)
    se_e = (early["abs_ci"][1] - early["abs_ci"][0]) / (2 * 1.96)
    se_l = (late["abs_ci"][1] - late["abs_ci"][0]) / (2 * 1.96)
    z = (early["abs_diff"] - late["abs_diff"]) / np.sqrt(se_e ** 2 + se_l ** 2)
    p_decay = float(stats.norm.sf(z))  # one-sided: early > late
    return {
        "early_lift": early["rel_lift"], "late_lift": late["rel_lift"],
        "early_p": early["p_value"], "late_p": late["p_value"],
        "p_decay": p_decay,
        "detected": early["significant"] and early["abs_diff"] > 0 and p_decay < 0.05,
    }


def daily_spikes(daily: pd.DataFrame) -> list[dict]:
    """Flag days where the purchase rate (both arms pooled) jumps away from
    its recent average by more than SPIKE_Z standard errors."""
    pooled = daily.groupby("exposure_day")[["purchases", "users"]].sum()
    pooled["rate"] = pooled["purchases"] / pooled["users"]
    baseline = pooled["rate"].rolling(ROLLING_WINDOW, min_periods=3).mean().shift(1)
    se = np.sqrt(baseline * (1 - baseline) / pooled["users"])
    pooled["z"] = (pooled["rate"] - baseline) / se
    spikes = pooled[pooled["z"].abs() > SPIKE_Z]
    return [{"day": int(day), "rate": float(r.rate), "baseline": float(baseline[day]), "z": float(r.z)}
            for day, r in spikes.iterrows()]


def detect_anomalies(df, trust, primary, guardrails, segments, daily) -> list[dict]:
    found = []

    srm = trust["srm"]
    if not srm["passed"]:
        found.append({"severity": "high", "title": "Sample ratio mismatch",
                      "detail": f"Groups are {srm['n_control']:,} control vs {srm['n_treatment']:,} treatment "
                                f"(planned 50/50, {fmt_p(srm['p_value'])}). Users are being lost or "
                                "double-counted, so results cannot be trusted."})
    if not trust["duplicates"]["passed"]:
        found.append({"severity": "high", "title": "Users in both groups",
                      "detail": f"{trust['duplicates']['users_in_both_arms']} users appear in both arms."})

    for g in guardrails:
        if g["status"] == "FAIL":
            found.append({"severity": "high", "title": f"Guardrail breached: {g['name']}",
                          "detail": f"{g['name']} changed by {g['rel_lift']:+.1%} "
                                    f"(tolerance {g['tolerance']:.0%}, {fmt_p(g['p_value'])})."})

    opposite = segments[segments["against_average"] & (segments["p_value"] < 0.05)]
    for _, s in opposite.iterrows():
        found.append({"severity": "medium", "title": f"Segment moves against the average: {s['label']}",
                      "detail": f"Overall lift is {primary['rel_lift']:+.1%} but this segment is "
                                f"{s['rel_lift']:+.1%} ({fmt_p(s['p_value'])}). Treat as a hypothesis."})

    nov = novelty_check(df)
    if nov["detected"]:
        found.append({"severity": "medium", "title": "Effect is fading (novelty effect)",
                      "detail": f"Lift was {nov['early_lift']:+.1%} in days {EARLY_DAYS[0]}-{EARLY_DAYS[1]} "
                                f"but {nov['late_lift']:+.1%} in days {LATE_DAYS[0]}-{LATE_DAYS[1]}. "
                                "The long-run effect is likely smaller than the headline."})

    for s in daily_spikes(daily):
        found.append({"severity": "info", "title": f"Daily spike on day {s['day']}",
                      "detail": f"Purchase rate (both groups) was {s['rate']:.2%} vs a recent average of "
                                f"{s['baseline']:.2%} (z = {s['z']:.1f}). Likely an outside event such as a "
                                "sale; it affects both groups equally."})
    return found
