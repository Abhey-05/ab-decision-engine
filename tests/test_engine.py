"""
Tests that prove the engine is trustworthy.

Run with:  pytest -q
"""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from engine import decision as D
from engine.analyst import build_results_json, template_readout, verify_numbers
from engine.checks import srm_check
from engine.data import EXPERIMENTS, generate_experiment, simulate
from engine.pipeline import analyze
from engine.stats import (benjamini_hochberg, guardrail_check, sample_size_per_arm,
                          two_proportion_ztest, welch_ttest)
from engine.uplift import fit_uplift


# --- fixtures: generate + analyse each experiment once per test session ------

@pytest.fixture(scope="session")
def data():
    return {k: generate_experiment(k) for k in EXPERIMENTS}


@pytest.fixture(scope="session")
def results(data):
    return {k: analyze(df) for k, df in data.items()}


# --- statistics --------------------------------------------------------------

def test_ztest_matches_statsmodels():
    ours = two_proportion_ztest(720, 60_000, 828, 60_000)
    z, p = proportions_ztest([828, 720], [60_000, 60_000])
    assert ours["z"] == pytest.approx(z, rel=1e-9)
    assert ours["p_value"] == pytest.approx(p, rel=1e-9)
    assert ours["rel_lift"] == pytest.approx(0.15)


def test_welch_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(1)
    a, b = rng.lognormal(0, 1, 500), rng.lognormal(0.1, 1, 700)
    ours = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert ours["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


def test_benjamini_hochberg_matches_statsmodels():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    adj, rej = benjamini_hochberg(p)
    ref_rej, ref_adj, _, _ = multipletests(p, method="fdr_bh")
    np.testing.assert_allclose(adj, ref_adj)
    assert list(rej) == list(ref_rej)


def test_sample_size_sensible():
    # 1.2% baseline, +15% MDE -> tens of thousands per arm
    n = sample_size_per_arm(0.012, 0.15)
    assert 50_000 < n < 70_000
    # halving the MDE roughly quadruples the sample size
    assert sample_size_per_arm(0.012, 0.075) / n == pytest.approx(4, rel=0.1)


def test_guardrail_states():
    assert guardrail_check(-0.20, (-0.27, -0.13), 1e-6, 0.03, "decrease")["status"] == "FAIL"
    assert guardrail_check(0.001, (-0.004, 0.006), 0.6, 0.05, "increase")["status"] == "PASS"
    assert guardrail_check(-0.03, (-0.12, 0.07), 0.5, 0.03, "decrease")["status"] == "WATCH"


# --- trust checks ------------------------------------------------------------

def test_srm_flags_61k_59k():
    assert not srm_check(61_000, 59_000)["passed"]


def test_srm_passes_normal_split():
    assert srm_check(60_090, 59_910)["passed"]


def test_aa_false_positive_rate(data):
    """A/A test: randomly split the CONTROL users of one experiment in two,
    500 times. There is no real difference, so ~5% of tests should come
    out 'significant' (that's what alpha = 0.05 means)."""
    y = data["price_breakup"].query("variant == 'control'")["purchased"].to_numpy()
    rng = np.random.default_rng(42)
    n_runs, hits = 500, 0
    for _ in range(n_runs):
        a = rng.random(len(y)) < 0.5
        r = two_proportion_ztest(int(y[a].sum()), int(a.sum()), int(y[~a].sum()), int((~a).sum()))
        hits += r["significant"]
    fpr = hits / n_runs
    assert 0.03 <= fpr <= 0.07, fpr


# --- planted effects are recovered --------------------------------------------

def test_price_breakup_effect_recovered(results):
    _, truth = simulate("price_breakup")
    true_lift = truth["purchase_treatment"] / truth["purchase_control"] - 1
    p = results["price_breakup"]["primary"]
    assert p["significant"]
    lo, hi = p["rel_ci"]
    assert lo <= true_lift <= hi
    assert abs(true_lift - 0.15) < 0.01  # we planted ~+15%


def test_broken_experiment_has_srm(data):
    counts = data["video_reviews"]["variant"].value_counts()
    assert counts["control"] == 61_000 and counts["treatment"] == 59_000


def test_uplift_beats_random(data):
    fit = fit_uplift(data["try_at_home"])
    assert fit["qini_t"] > 0
    assert fit["curve_t"]["model"].max() > fit["curve_t"]["random"].max()


# --- decisions ------------------------------------------------------------------

@pytest.mark.parametrize("key, expected", [
    ("price_breakup", D.SHIP),
    ("try_at_home", D.TARGETED),
    ("discount_popup", D.ITERATE),
    ("video_reviews", D.NO_DECISION),
])
def test_decisions(results, key, expected):
    assert results[key]["decision"]["decision"] == expected


def test_novelty_detected_in_popup_only(results):
    assert results["discount_popup"]["novelty"]["detected"]
    assert not results["price_breakup"]["novelty"]["detected"]


# --- AI analyst number checker ------------------------------------------------

def test_template_readout_is_fully_verified(results):
    for key, res in results.items():
        facts = build_results_json(EXPERIMENTS[key], res)
        assert verify_numbers(template_readout(facts), facts)["all_verified"], key


def test_number_checker_catches_wrong_number(results):
    facts = build_results_json(EXPERIMENTS["price_breakup"], results["price_breakup"])
    lift = facts["primary_metric"]["relative_lift_pct"]
    good = f"Purchases rose {lift}% with 95% confidence."
    bad = good + " Revenue will grow by 37.4% next quarter."
    assert verify_numbers(good, facts)["all_verified"]
    check = verify_numbers(bad, facts)
    assert not check["all_verified"] and "37.4" in check["unverified"]


def test_number_checker_allows_rounding(results):
    facts = build_results_json(EXPERIMENTS["price_breakup"], results["price_breakup"])
    lift = facts["primary_metric"]["relative_lift_pct"]
    assert verify_numbers(f"about {round(lift)}%", facts)["all_verified"]


def test_facts_are_json_serialisable(results):
    import json
    for key, res in results.items():
        json.dumps(build_results_json(EXPERIMENTS[key], res))
