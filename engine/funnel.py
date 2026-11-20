"""Conversion funnel: view -> add to cart -> purchase, per variant."""

from __future__ import annotations

import pandas as pd

STEPS = [("viewed_product", "Viewed product"),
         ("added_to_cart", "Added to cart"),
         ("purchased", "Purchased")]


def funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, step) with:
    - users: how many reached the step
    - overall_rate: share of ALL assigned users who reached it
    - step_rate: share of users from the PREVIOUS step who made it here
      (shows *where* in the journey the treatment helps or hurts)
    """
    rows = []
    for variant, g in df.groupby("variant"):
        total = len(g)
        prev = total
        for col, label in STEPS:
            reached = int(g[col].sum())
            rows.append({
                "variant": variant, "step": label, "users": reached,
                "overall_rate": reached / total,
                "step_rate": reached / prev if prev else 0.0,
            })
            prev = reached
    return pd.DataFrame(rows)
