"""
AI Product Analyst.

The golden rule: the AI never calculates anything. All numbers come from the
statistics engine, get packed into a JSON "fact sheet", and the AI is told to
explain ONLY those numbers. Then we check its work: every number in the
AI's text must appear in the JSON (allowing for rounding). If anything
doesn't match, the app shows a warning listing the unverified numbers.

If there is no API key, or the call fails, a deterministic template writes
the same readout from the same JSON - clearly labelled as such.
"""

from __future__ import annotations

import json
import re

# Google Gemini. Change the default model here, in one place, or override it
# with a GEMINI_MODEL secret / environment variable (see ui.get_ai_settings).
MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 8192   # generous: some Gemini models "think" before writing

SYSTEM_PROMPT = """You are a senior product analyst at an online jewellery company. You explain A/B test results to product managers and non-technical stakeholders.

Rules:
- Use ONLY the numbers that appear in the JSON you are given. Copy them exactly as written (you may drop trailing decimals, e.g. 15.0 -> 15). Never calculate new numbers: no differences, sums, ratios, conversions or projections.
- If something is not in the JSON, say it is unknown rather than guessing.
- If trust_checks.passed is false, say clearly that the results cannot be trusted and must not be acted on.
- Plain English, short sentences, no jargon without a brief explanation.

Output exactly this markdown template and nothing else:

### Headline
One sentence.

### Recommendation
The decision from the JSON in bold, then one or two sentences on why.

### Evidence
3-5 bullet points with the key numbers.

### Risks & anomalies
Bullet points covering guardrails, anomalies and caveats.

### Hypotheses to test next
Exactly 3 numbered hypotheses, each in this format:
We believe [change] for [segment] will [outcome] because [evidence]; measured by [metric]."""


# ---------------------------------------------------------------------------
# 1. Build the fact sheet
# ---------------------------------------------------------------------------

def _pct(x: float, nd: int = 1) -> float:
    return round(float(x) * 100, nd)


def _p(p: float) -> float | str:
    """p-values: 4 decimals, or '<0.0001' for tiny ones (easier to quote)."""
    return "<0.0001" if p < 0.0001 else round(float(p), 4)


def build_results_json(meta: dict, res: dict, uplift: dict | None = None) -> dict:
    """Turn the engine's output into a compact, rounded JSON fact sheet."""
    pr = res["primary"]
    seg = res["segments"]
    nov = res["novelty"]
    out = {
        "experiment": meta["name"],
        "change_tested": meta["change"],
        "confidence_level_pct": 95,
        "significance_threshold_p": 0.05,
        "users": {"control": pr["n_control"], "treatment": pr["n_treatment"]},
        "trust_checks": {
            "passed": bool(res["trust"]["passed"]),
            "sample_ratio_mismatch_p": _p(res["trust"]["srm"]["p_value"]),
            "srm_threshold_p": 0.001,
            "users_in_both_groups": res["trust"]["duplicates"]["users_in_both_arms"],
        },
        "primary_metric": {
            "name": "Purchase conversion",
            "control_pct": _pct(pr["control"], 2), "treatment_pct": _pct(pr["treatment"], 2),
            "relative_lift_pct": _pct(pr["rel_lift"]),
            "ci_low_pct": _pct(pr["rel_ci"][0]), "ci_high_pct": _pct(pr["rel_ci"][1]),
            "p_value": _p(pr["p_value"]), "significant": bool(pr["significant"]),
        },
        "secondary_metrics": [{
            "name": m["name"], "relative_change_pct": _pct(m["rel_lift"]),
            "ci_low_pct": _pct(m["rel_ci"][0]), "ci_high_pct": _pct(m["rel_ci"][1]),
            "p_value_adjusted": _p(m["p_adj"]),
        } for m in res["secondary"]],
        "guardrails": [{
            "name": g["name"], "relative_change_pct": _pct(g["rel_lift"]),
            "ci_low_pct": _pct(g["rel_ci"][0]), "ci_high_pct": _pct(g["rel_ci"][1]),
            "p_value": _p(g["p_value"]), "tolerance_pct": int(round(g["tolerance"] * 100)),
            "status": g["status"],
        } for g in res["guardrails"]],
        "segments_tested": int(len(seg)),
        "best_segments": _segments(seg.head(3)),
        "worst_segments": _segments(seg.tail(3).iloc[::-1]),
        "novelty_check": {
            "early_days": "1-4", "late_days": "8-14",
            "early_lift_pct": _pct(nov["early_lift"]), "late_lift_pct": _pct(nov["late_lift"]),
            "fading_detected": bool(nov["detected"]),
        },
        "anomalies": [f"{a['title']}: {a['detail']}" for a in res["anomalies"]],
        "decision": res["decision"]["decision"],
        "decision_reasons": res["decision"]["reasons"],
    }
    if uplift is not None:
        out["uplift_model"] = uplift
    return out


def _segments(df) -> list[dict]:
    return [{"segment": r["label"], "relative_lift_pct": _pct(r["rel_lift"]),
             "ci_low_pct": _pct(r["ci_low"]), "ci_high_pct": _pct(r["ci_high"]),
             "p_value_adjusted": _p(r["p_adj"])} for _, r in df.iterrows()]


# ---------------------------------------------------------------------------
# 2. Number verification
# ---------------------------------------------------------------------------

# Matches 61,000 / 1.38 / 15 / 0.0065 (signs are ignored on purpose; we
# compare absolute values so "-3.3%" and "down 3.3%" are treated the same).
NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|#+)\s", re.MULTILINE)


def _numbers_in_text(text: str) -> list[str]:
    text = LIST_MARKER_RE.sub(" ", text)          # ignore "1. " list numbering
    return NUMBER_RE.findall(text)


def _numbers_in_json(obj) -> set[float]:
    """Every number in the JSON, including numbers inside strings."""
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(abs(float(obj)))
    elif isinstance(obj, str):
        found |= {float(n.replace(",", "")) for n in NUMBER_RE.findall(obj)}
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _numbers_in_json(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _numbers_in_json(v)
    return found


def _matches(token: str, allowed: set[float]) -> bool:
    """Is this number in the JSON, allowing for rounding?
    '15' matches 15.0 or 14.96; '1.3' matches 1.33; '61,000' matches 61000.
    Large whole numbers may be rounded by up to 1% ('about 61,000')."""
    x = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    for v in allowed:
        if abs(round(v, decimals) - x) < 1e-9:
            return True
        if decimals == 0 and x >= 100 and abs(x - v) <= 0.01 * v:
            return True
    return False


def verify_numbers(text: str, facts: dict) -> dict:
    allowed = _numbers_in_json(facts)
    tokens = _numbers_in_text(text)
    unverified = sorted({t for t in tokens if not _matches(t, allowed)})
    return {"checked": len(tokens), "unverified": unverified, "all_verified": not unverified}


# ---------------------------------------------------------------------------
# 3. Generate the readout (Gemini, or template fallback)
# ---------------------------------------------------------------------------

def template_readout(f: dict) -> str:
    """Deterministic readout built from the same JSON. Used when the AI is
    unavailable, and as a reference for what a good readout contains."""
    p = f["primary_metric"]
    lines = ["### Headline"]
    if not f["trust_checks"]["passed"]:
        lines.append(f"The {f['experiment']} test is broken, so its results cannot be trusted.")
    else:
        word = "increased" if p["relative_lift_pct"] > 0 else "decreased"
        sure = "a real effect" if p["significant"] else "not distinguishable from luck"
        lines.append(f"{f['experiment']} {word} purchase conversion by {abs(p['relative_lift_pct'])}% ({sure}).")

    lines += ["", "### Recommendation", f"**{f['decision']}.** {f['decision_reasons'][-1]}", "", "### Evidence"]
    lines.append(f"- Users: {f['users']['control']:,} control vs {f['users']['treatment']:,} treatment.")
    lines.append(f"- Purchase conversion {p['control_pct']}% vs {p['treatment_pct']}% "
                 f"({p['relative_lift_pct']:+}%, 95% CI {p['ci_low_pct']}% to {p['ci_high_pct']}%, p = {p['p_value']}).")
    for m in f["secondary_metrics"]:
        lines.append(f"- {m['name']}: {m['relative_change_pct']:+}% (adjusted p = {m['p_value_adjusted']}).")
    b = f["best_segments"][0]
    lines.append(f"- Strongest segment: {b['segment']} at {b['relative_lift_pct']:+}%.")

    lines += ["", "### Risks & anomalies"]
    for g in f["guardrails"]:
        lines.append(f"- Guardrail {g['name']}: {g['relative_change_pct']:+}% vs tolerance {g['tolerance_pct']}% -> {g['status']}.")
    for a in f["anomalies"]:
        lines.append(f"- {a}")
    if f["segments_tested"] > 5:
        lines.append(f"- {f['segments_tested']} segments were tested, so segment results are hypotheses, not conclusions.")

    lines += ["", "### Hypotheses to test next"]
    best, worst = f["best_segments"], f["worst_segments"]
    lines.append(f"1. We believe {f['change_tested'].lower()} for {best[0]['segment']} will raise purchases "
                 f"because this segment showed {best[0]['relative_lift_pct']:+}%; measured by purchase conversion.")
    lines.append(f"2. We believe a variant designed for {worst[0]['segment']} will avoid losses there "
                 f"because this segment showed {worst[0]['relative_lift_pct']:+}%; measured by purchase conversion.")
    g = max(f["guardrails"], key=lambda g: abs(g["relative_change_pct"]))
    lines.append(f"3. We believe adjusting the design to protect {g['name'].lower()} for all users will keep it within "
                 f"{g['tolerance_pct']}% because it moved {g['relative_change_pct']:+}% in this test; "
                 f"measured by {g['name'].lower()}.")
    return "\n".join(lines)


def _call_gemini(facts: dict, api_key: str, model: str) -> str:
    """Send the fact sheet to Gemini and return its text (may be empty)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))  # ms
    resp = client.models.generate_content(
        model=model,
        contents="Experiment results JSON:\n```json\n" + json.dumps(facts, indent=2) + "\n```\nWrite the readout.",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,              # low = consistent, factual wording
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return (resp.text or "").strip()


def generate_readout(facts: dict, api_key: str | None, model: str = MODEL) -> dict:
    """Returns {text, source, model, note, verification}. source is 'gemini' or 'template'."""
    note = None
    if api_key:
        try:
            text = _call_gemini(facts, api_key, model)
            if text:
                return {"text": text, "source": "gemini", "model": model, "note": None,
                        "verification": verify_numbers(text, facts)}
            note = "The AI returned an empty response (possibly blocked by a safety filter)."
        except Exception as e:  # any failure -> fall back, never crash the demo
            note = f"AI call failed ({type(e).__name__}); showing the template readout instead."
    else:
        note = "No API key configured; showing the template readout."

    text = template_readout(facts)
    return {"text": text, "source": "template", "model": None, "note": note,
            "verification": verify_numbers(text, facts)}
