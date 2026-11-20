"""
Decision engine: explicit, readable rules -> one recommendation + reasons.

Order matters - the first matching rule wins:
  1. Trust checks failed                  -> No decision (fix the test)
  2. Primary metric significantly up:
       a guardrail FAILED                 -> Iterate
       a segment significantly hurt       -> Targeted rollout (exclude it)
       otherwise                          -> Ship to all
  3. Primary metric significantly down    -> Don't ship
  4. Primary flat:
       some segment significantly up      -> Targeted rollout
       otherwise                          -> Don't ship
"Significant" for segments uses Benjamini-Hochberg adjusted p < 0.05,
because we tested many segments.
"""

from __future__ import annotations

import pandas as pd

from .stats import fmt_p

SHIP = "Ship to all"
TARGETED = "Targeted rollout"
ITERATE = "Iterate"
DONT_SHIP = "Don't ship"
NO_DECISION = "No decision (fix the test)"


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def decide(trust: dict, primary: dict, guardrails: list[dict], segments: pd.DataFrame,
           anomalies: list[dict] | None = None) -> dict:
    reasons: list[str] = []
    anomalies = anomalies or []

    # Rule 1: never read results from a broken test.
    if not trust["passed"]:
        srm = trust["srm"]
        if not srm["passed"]:
            reasons.append(f"Sample ratio mismatch: {srm['n_control']:,} vs {srm['n_treatment']:,} users "
                           f"when we planned 50/50 ({fmt_p(srm['p_value'])}). This is almost impossible by chance.")
        if not trust["duplicates"]["passed"]:
            reasons.append("Some users appear in both groups.")
        for b in trust["balance"]:
            if not b["passed"]:
                reasons.append(f"Groups differ before the test started on {b['attribute']}.")
        reasons.append("Any lift shown is likely caused by the bug, not the feature. Fix the logging and re-run.")
        return {"decision": NO_DECISION, "reasons": reasons}

    lift_txt = (f"Purchase conversion {primary['control']:.2%} -> {primary['treatment']:.2%} "
                f"({_pct(primary['rel_lift'])}, 95% CI {_pct(primary['rel_ci'][0])} to {_pct(primary['rel_ci'][1])}, "
                f"{fmt_p(primary['p_value'])})")
    failed = [g for g in guardrails if g["status"] == "FAIL"]
    watch = [g for g in guardrails if g["status"] == "WATCH"]
    sig = segments[segments["significant_adj"]]
    seg_up = sig[sig["rel_lift"] > 0]
    seg_down = sig[sig["rel_lift"] < 0]
    fading = [a for a in anomalies if "novelty" in a["title"].lower()]

    if primary["significant"] and primary["rel_lift"] > 0:
        reasons.append(lift_txt + " - a real improvement, not luck.")
        if failed:
            for g in failed:
                reasons.append(f"Guardrail failed: {g['name']} {_pct(g['rel_lift'])} "
                               f"(we only tolerate {g['tolerance']:.0%}).")
            for _, s in seg_down.iterrows():
                reasons.append(f"Hurts {s['label']} ({_pct(s['rel_lift'])}): these users would have bought anyway.")
            if fading:
                reasons.append("The effect is fading over time, so the long-run gain is smaller.")
            reasons.append("Keep the idea but redesign it to protect the guardrails, then re-test.")
            return {"decision": ITERATE, "reasons": reasons}
        if len(seg_down):
            for _, s in seg_down.iterrows():
                reasons.append(f"But it hurts {s['label']} ({_pct(s['rel_lift'])}) - exclude this group.")
            return {"decision": TARGETED, "reasons": reasons}
        reasons.append("No guardrail shows significant harm"
                       + (f"; {', '.join(g['name'] for g in watch)} too noisy to prove safe - monitor after launch."
                          if watch else "."))
        if len(seg_up):
            top = seg_up.iloc[0]
            reasons.append(f"Biggest gains in {top['label']} ({_pct(top['rel_lift'])}); no segment is hurt.")
        return {"decision": SHIP, "reasons": reasons}

    if primary["significant"] and primary["rel_lift"] < 0:
        reasons.append(lift_txt + " - it made things worse.")
        return {"decision": DONT_SHIP, "reasons": reasons}

    # Primary is flat.
    reasons.append(lift_txt + " - no clear effect on average.")
    if len(seg_up):
        names = ", ".join(f"{s['label']} ({_pct(s['rel_lift'])})" for _, s in seg_up.head(3).iterrows())
        reasons.append(f"But it clearly helps some groups: {names}.")
        if len(segments[(segments["rel_lift"] < 0)]):
            reasons.append("Other groups are flat or slightly negative, which cancels out the average.")
        reasons.append("Roll out only to the groups that benefit and confirm with a follow-up test.")
        return {"decision": TARGETED, "reasons": reasons}
    reasons.append("No segment shows a reliable benefit either.")
    return {"decision": DONT_SHIP, "reasons": reasons}
