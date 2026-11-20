"""
Trust checks: run BEFORE looking at results.

If the experiment itself is broken, its results are meaningless no matter
how good they look. These checks catch the most common breakages.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .stats import two_proportion_ztest, welch_ttest

SRM_THRESHOLD = 0.001      # industry-standard: only flag very unlikely splits
BALANCE_THRESHOLD = 0.001


def srm_check(n_control: int, n_treatment: int, expected_share: float = 0.5) -> dict:
    """Sample Ratio Mismatch: did users land in each group in the planned ratio?

    We planned 50/50. With 120,000 fair coin flips you'd expect a split close
    to 60,000/60,000 - off by a few hundred at most. A chi-square test tells
    us how surprising the observed split is. A 61,000 / 59,000 split has a
    p-value around 1e-8: essentially impossible by chance, so something in the
    pipeline is dropping or duplicating users.
    """
    total = n_control + n_treatment
    expected = [total * expected_share, total * (1 - expected_share)]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {
        "n_control": int(n_control), "n_treatment": int(n_treatment),
        "share_treatment": n_treatment / total,
        "chi2": float(chi2), "p_value": float(p),
        "passed": p >= SRM_THRESHOLD,
    }


def duplicate_check(df: pd.DataFrame) -> dict:
    """Users that appear in BOTH arms (or twice) contaminate the comparison."""
    arms_per_user = df.groupby("user_id")["variant"].nunique()
    in_both = int((arms_per_user > 1).sum())
    repeated = int(df["user_id"].duplicated().sum())
    return {"users_in_both_arms": in_both, "duplicate_rows": repeated,
            "passed": in_both == 0 and repeated == 0}


def balance_check(df: pd.DataFrame) -> list[dict]:
    """Pre-experiment balance: groups should look alike on things the
    treatment cannot have changed (measured before exposure).
    """
    c = df[df["variant"] == "control"]
    t = df[df["variant"] == "treatment"]
    rows = []

    r = welch_ttest(c["prior_sessions"], t["prior_sessions"])
    rows.append({"attribute": "Prior sessions (average)", "control": r["control"],
                 "treatment": r["treatment"], "p_value": r["p_value"]})

    r = two_proportion_ztest(int(c["is_new_user"].sum()), len(c), int(t["is_new_user"].sum()), len(t))
    rows.append({"attribute": "New users (share)", "control": r["control"],
                 "treatment": r["treatment"], "p_value": r["p_value"]})

    for row in rows:
        row["passed"] = row["p_value"] >= BALANCE_THRESHOLD
    return rows


def run_trust_checks(df: pd.DataFrame) -> dict:
    counts = df["variant"].value_counts()
    srm = srm_check(int(counts.get("control", 0)), int(counts.get("treatment", 0)))
    dup = duplicate_check(df)
    bal = balance_check(df)
    return {
        "srm": srm,
        "duplicates": dup,
        "balance": bal,
        "passed": srm["passed"] and dup["passed"] and all(b["passed"] for b in bal),
    }
