"""
Compute every metric of an experiment in one place.

Primary metric   : purchase conversion (purchases / users) - the decision metric.
Secondary metrics: product view rate, add-to-cart rate, revenue per user.
                   Benjamini-Hochberg is applied across these.
Guardrails       : average order value, return rate, page load time -
                   things that must not get meaningfully worse.
"""

from __future__ import annotations

import pandas as pd

from .stats import (DEFAULT_GUARDRAILS, benjamini_hochberg, bootstrap_ci,
                    guardrail_check, two_proportion_ztest, welch_ttest)


def _split(df: pd.DataFrame):
    return df[df["variant"] == "control"], df[df["variant"] == "treatment"]


def _prop(c, t, col):
    return two_proportion_ztest(int(c[col].sum()), len(c), int(t[col].sum()), len(t))


def primary_metric(df: pd.DataFrame) -> dict:
    c, t = _split(df)
    r = _prop(c, t, "purchased")
    r.update(name="Purchase conversion", kind="primary", unit="rate")
    return r


def secondary_metrics(df: pd.DataFrame) -> list[dict]:
    c, t = _split(df)
    out = []
    for col, name in [("viewed_product", "Product view rate"), ("added_to_cart", "Add-to-cart rate")]:
        r = _prop(c, t, col)
        r.update(name=name, kind="secondary", unit="rate")
        out.append(r)
    r = welch_ttest(c["order_value"], t["order_value"])
    r.update(name="Revenue per user", kind="secondary", unit="rupees")
    out.append(r)

    adj, _ = benjamini_hochberg([m["p_value"] for m in out])
    for m, p in zip(out, adj):
        m["p_adj"] = float(p)
        m["significant_adj"] = p < 0.05
    return out


def guardrail_metrics(df: pd.DataFrame, guardrails: dict | None = None) -> list[dict]:
    """Guardrails with pass / watch / fail status against tolerances."""
    guardrails = guardrails or DEFAULT_GUARDRAILS
    c, t = _split(df)
    c_buy, t_buy = c[c["purchased"] == 1], t[t["purchased"] == 1]
    out = []

    # Average order value: skewed, so Welch for the p-value + bootstrap CI.
    r = welch_ttest(c_buy["order_value"], t_buy["order_value"])
    boot = bootstrap_ci(c_buy["order_value"].to_numpy(), t_buy["order_value"].to_numpy())
    r["rel_ci"], r["abs_ci"], r["ci_method"] = boot["rel_ci"], boot["abs_ci"], "bootstrap"
    r.update(name="Average order value", unit="rupees")
    out.append(r)

    # Return rate: returned orders / orders.
    r = _prop(c_buy, t_buy, "returned")
    r.update(name="Return rate", unit="rate")
    out.append(r)

    # Page load: huge samples, Welch is fine.
    r = welch_ttest(c["page_load_ms"], t["page_load_ms"])
    r.update(name="Page load time", unit="ms")
    out.append(r)

    for m in out:
        cfg = guardrails[m["name"]]
        m["kind"] = "guardrail"
        m.update(guardrail_check(m["rel_lift"], m["rel_ci"], m["p_value"],
                                 cfg["tolerance"], cfg["bad_direction"]))
        m["bad_direction"] = cfg["bad_direction"]
    return out


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase rate per exposure day and variant (for charts and anomaly checks)."""
    g = df.groupby(["exposure_day", "variant"])["purchased"].agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    return g.rename(columns={"sum": "purchases", "count": "users"})
