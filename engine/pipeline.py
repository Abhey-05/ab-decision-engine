"""
The full analysis in one call: trust checks -> metrics -> funnel -> segments
-> anomalies -> decision. The Streamlit app caches the result of `analyze`.
(Uplift modelling is separate because it is slower; see uplift.py.)
"""

from __future__ import annotations

import pandas as pd

from .anomalies import detect_anomalies, novelty_check
from .checks import run_trust_checks
from .decision import decide
from .funnel import funnel_table
from .metrics import daily_series, guardrail_metrics, primary_metric, secondary_metrics
from .segments import segment_lifts


def analyze(df: pd.DataFrame) -> dict:
    trust = run_trust_checks(df)
    primary = primary_metric(df)
    secondary = secondary_metrics(df)
    guardrails = guardrail_metrics(df)
    funnel = funnel_table(df)
    segments = segment_lifts(df)
    daily = daily_series(df)
    anomalies = detect_anomalies(df, trust, primary, guardrails, segments, daily)
    decision = decide(trust, primary, guardrails, segments, anomalies)
    return {
        "trust": trust, "primary": primary, "secondary": secondary,
        "guardrails": guardrails, "funnel": funnel, "segments": segments,
        "daily": daily, "novelty": novelty_check(df),
        "anomalies": anomalies, "decision": decision,
    }
