"""
Shared Streamlit helpers: cached data/analysis, the experiment picker,
colours, and small layout pieces used by every page.

The engine (engine/) has no Streamlit code; this file is the bridge.
"""

from __future__ import annotations

import os

import streamlit as st

from engine.analyst import MODEL, build_results_json
from engine.data import EXPERIMENTS, generate_experiment
from engine.pipeline import analyze
from engine.uplift import fit_uplift, uplift_facts

# Consistent colours everywhere: control = grey, treatment = blue.
CONTROL = "#6B7280"
TREATMENT = "#2563EB"
GOOD = "#15803D"
BAD = "#B91C1C"
WARN = "#B45309"
GRID = "#E5E7EB"
ARM_COLORS = {"control": CONTROL, "treatment": TREATMENT}

DECISION_STYLE = {   # decision -> (colour, icon)
    "Ship to all": (GOOD, "🚀"),
    "Targeted rollout": (TREATMENT, "🎯"),
    "Iterate": (WARN, "🔁"),
    "Don't ship": (BAD, "✋"),
    "No decision (fix the test)": (BAD, "🛑"),
}
STATUS_STYLE = {"PASS": (GOOD, "✅ Pass"), "WATCH": (WARN, "👀 Watch"), "FAIL": (BAD, "❌ Fail")}


# ---------------------------------------------------------------------------
# Cached computation (runs once per experiment per server, then instant)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating experiment data...")
def get_data(key: str):
    return generate_experiment(key)


@st.cache_data(show_spinner="Running the statistics engine...")
def get_results(key: str) -> dict:
    return analyze(get_data(key))


@st.cache_data(show_spinner="Training uplift models (about 5 seconds, once)...")
def get_uplift(key: str) -> dict:
    return fit_uplift(get_data(key))


def get_facts(key: str, include_uplift: bool = True) -> dict:
    up = uplift_facts(get_uplift(key)) if include_uplift else None
    return build_results_json(EXPERIMENTS[key], get_results(key), up)


def _setting(name: str) -> str | None:
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:          # no secrets.toml at all
        value = None
    return (value or os.environ.get(name) or "").strip() or None


def get_ai_settings() -> tuple[str | None, str]:
    """(Gemini API key, model name). The key is never hard-coded."""
    return _setting("GEMINI_API_KEY"), _setting("GEMINI_MODEL") or MODEL


# ---------------------------------------------------------------------------
# Sidebar experiment picker (drives every page)
# ---------------------------------------------------------------------------

def experiment_picker() -> str:
    keys = list(EXPERIMENTS)
    st.session_state.setdefault("experiment", keys[0])
    st.sidebar.selectbox(
        "Experiment", keys, key="experiment",
        format_func=lambda k: EXPERIMENTS[k]["name"],
        help="Every page shows results for the experiment picked here.",
    )
    key = st.session_state["experiment"]
    st.sidebar.caption(f"**Change tested:** {EXPERIMENTS[key]['change']}.")
    decision = get_results(key)["decision"]["decision"]
    color, icon = DECISION_STYLE[decision]
    st.sidebar.markdown(f"Recommendation: <span style='color:{color};font-weight:600'>{icon} {decision}</span>",
                        unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.caption("Data is synthetic, generated with known built-in effects so the methods can be verified.")
    return key


def current() -> str:
    return st.session_state.get("experiment", list(EXPERIMENTS)[0])


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def header(title: str, one_liner: str, explainer: str) -> None:
    """Every page: title, one plain-language sentence, and a beginner explainer."""
    st.title(title)
    st.markdown(f"<p style='font-size:1.1rem;color:#374151'>{one_liner}</p>", unsafe_allow_html=True)
    with st.expander("💡 What does this mean?"):
        st.markdown(explainer)


def decision_card(decision: str, reasons: list[str], n: int = 3) -> None:
    color, icon = DECISION_STYLE[decision]
    items = "".join(f"<li style='margin-bottom:4px'>{r}</li>" for r in reasons[:n])
    st.markdown(
        f"""<div style="border:1px solid {color}55;border-left:6px solid {color};border-radius:10px;
        padding:16px 20px;background:{color}0D">
        <div style="font-size:0.85rem;color:#6B7280;text-transform:uppercase;letter-spacing:.05em">Recommendation</div>
        <div style="font-size:1.6rem;font-weight:700;color:{color};margin:4px 0 8px">{icon} {decision}</div>
        <ul style="margin:0;padding-left:20px;color:#1F2937">{items}</ul></div>""",
        unsafe_allow_html=True)


def blocked_banner(trust: dict) -> None:
    srm = trust["srm"]
    st.markdown(
        f"""<div style="border-radius:10px;padding:18px 22px;background:{BAD};color:white">
        <div style="font-size:1.4rem;font-weight:700">🛑 Results blocked: this test is broken</div>
        <div style="margin-top:6px;font-size:1.02rem">We planned a 50/50 split, but got
        <b>{srm['n_control']:,}</b> control vs <b>{srm['n_treatment']:,}</b> treatment users.
        With fair coin flips, a gap this big would happen less than once in a million tests.
        Something is losing users from one group (here: treatment users on slow pages never got logged),
        so the groups are no longer comparable and any "lift" may just be the bug.
        <b>Fix the logging and re-run before reading any result.</b></div></div>""",
        unsafe_allow_html=True)


def pct(x: float, nd: int = 1, sign: bool = True) -> str:
    return f"{x:+.{nd}%}" if sign else f"{x:.{nd}%}"


def plotly_layout(fig, height: int = 380, **kw):
    """Calm, consistent chart styling."""
    fig.update_layout(
        height=height, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#1F2937"),
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#9CA3AF")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#9CA3AF")
    return fig
