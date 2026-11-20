"""
Core statistics. Every number the app shows is computed here (never by the AI).

Plain-language summary of each tool:
- two_proportion_ztest: "Is the difference in conversion rates bigger than
  what coin-toss luck would produce?"
- welch_ttest / bootstrap_ci: the same question for averages (order value,
  page load), where values are spread out and skewed.
- sample_size_per_arm: "How many users do I need before I start?"
- benjamini_hochberg: "I checked many metrics; adjust so I don't get fooled
  by the one that looks good by chance."
- guardrail_check: "Did we break something we care about, beyond what we
  agreed to tolerate?"
"""

from __future__ import annotations

import numpy as np
from scipy import stats

ALPHA = 0.05  # 5% false-alarm rate; gives 95% confidence intervals


def fmt_p(p: float) -> str:
    """Readable p-value: 'p < 0.0001' for tiny values, otherwise 'p = 0.0123'."""
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def _z(alpha: float) -> float:
    """Two-sided critical value, e.g. 1.96 for alpha = 0.05."""
    return float(stats.norm.ppf(1 - alpha / 2))


# ---------------------------------------------------------------------------
# Conversion metrics (yes/no outcomes)
# ---------------------------------------------------------------------------

def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = ALPHA) -> dict:
    """Compare two conversion rates (x conversions out of n users).

    - The p-value uses the *pooled* standard error (standard textbook test,
      same as statsmodels.proportions_ztest), because under the "no effect"
      assumption both groups share one rate.
    - The confidence interval uses the *unpooled* standard error, because
      there we are describing the actual difference.
    - Relative lift CI = absolute CI / control rate (simple approximation that
      ignores the small uncertainty in the control rate itself).
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c

    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    lo, hi = diff - _z(alpha) * se, diff + _z(alpha) * se
    rel = diff / p_c if p_c > 0 else np.nan
    return {
        "control": p_c, "treatment": p_t,
        "n_control": n_c, "n_treatment": n_t,
        "abs_diff": diff, "abs_ci": (lo, hi),
        "rel_lift": rel,
        "rel_ci": (lo / p_c, hi / p_c) if p_c > 0 else (np.nan, np.nan),
        "z": float(z), "p_value": p_value,
        "significant": p_value < alpha,
    }


# ---------------------------------------------------------------------------
# Continuous metrics (order value, page load)
# ---------------------------------------------------------------------------

def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = ALPHA) -> dict:
    """Compare two means without assuming equal variances (Welch's t-test)."""
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    m_c, m_t = control.mean(), treatment.mean()
    v_c, v_t = control.var(ddof=1) / len(control), treatment.var(ddof=1) / len(treatment)
    se = np.sqrt(v_c + v_t)
    # Welch-Satterthwaite degrees of freedom
    dof = (v_c + v_t) ** 2 / (v_c ** 2 / (len(control) - 1) + v_t ** 2 / (len(treatment) - 1))
    t_stat = (m_t - m_c) / se
    p_value = float(2 * stats.t.sf(abs(t_stat), dof))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    diff = m_t - m_c
    lo, hi = diff - crit * se, diff + crit * se
    return {
        "control": m_c, "treatment": m_t,
        "n_control": len(control), "n_treatment": len(treatment),
        "abs_diff": diff, "abs_ci": (lo, hi),
        "rel_lift": diff / m_c, "rel_ci": (lo / m_c, hi / m_c),
        "t": float(t_stat), "p_value": p_value,
        "significant": p_value < alpha,
    }


def bootstrap_ci(control: np.ndarray, treatment: np.ndarray, n_boot: int = 2000,
                 alpha: float = ALPHA, seed: int = 0) -> dict:
    """Bootstrap CI for the difference (and ratio) of means.

    Idea: pretend our sample *is* the population, re-draw it with replacement
    2,000 times, and see how much the difference wobbles. The middle 95% of
    those re-draws is the confidence interval. No normality assumption, which
    matters for skewed things like jewellery order values.

    Vectorised: one (n_boot x n) matrix of random indices per arm instead of
    a Python loop. Only use on modest n (e.g. orders, not all users).
    """
    rng = np.random.default_rng(seed)
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    mc = control[rng.integers(0, len(control), (n_boot, len(control)))].mean(axis=1)
    mt = treatment[rng.integers(0, len(treatment), (n_boot, len(treatment)))].mean(axis=1)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "abs_ci": tuple(np.percentile(mt - mc, q)),
        "rel_ci": tuple(np.percentile(mt / mc - 1, q)),
    }


# ---------------------------------------------------------------------------
# Planning and multiple testing
# ---------------------------------------------------------------------------

def sample_size_per_arm(baseline: float, mde_rel: float, alpha: float = ALPHA, power: float = 0.8) -> int:
    """Users needed in EACH group to detect a relative lift of `mde_rel`.

    Standard two-proportion formula. Intuition: the smaller the baseline rate
    and the smaller the change you want to detect, the more users you need
    (it grows with 1 / effect^2).
    """
    p1 = baseline
    p2 = baseline * (1 + mde_rel)
    p_bar = (p1 + p2) / 2
    z_a = _z(alpha)
    z_b = float(stats.norm.ppf(power))
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def benjamini_hochberg(p_values, q: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg adjustment. Returns (adjusted p-values, reject flags).

    Sort p-values; the k-th smallest is compared to q * k / m. Adjusted
    p-values are made monotone so that "adjusted p < q" == "rejected".
    """
    p = np.asarray(p_values, float)
    m = len(p)
    if m == 0:
        return p, p.astype(bool)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    return adj, adj < q


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

# How much harm we accept, as a relative change. "increase" means a rise is bad.
DEFAULT_GUARDRAILS = {
    "Average order value": {"tolerance": 0.03, "bad_direction": "decrease"},
    "Return rate": {"tolerance": 0.05, "bad_direction": "increase"},
    "Page load time": {"tolerance": 0.05, "bad_direction": "increase"},
}


def guardrail_check(rel_change: float, rel_ci: tuple[float, float], p_value: float,
                    tolerance: float, bad_direction: str, alpha: float = ALPHA) -> dict:
    """Non-inferiority style check with three outcomes.

    harm = the change in the bad direction (positive = worse).
    - PASS  : even the worst end of the 95% CI is within tolerance
              ("we're confident it didn't get worse by more than X%").
    - FAIL  : the change is worse than tolerance AND statistically
              significant ("we're confident it got worse").
    - WATCH : neither — too noisy to prove safe, no evidence of harm.
    """
    lo, hi = rel_ci
    if bad_direction == "increase":
        harm, worst_case = rel_change, hi
    else:
        harm, worst_case = -rel_change, -lo

    if worst_case <= tolerance:
        status = "PASS"
    elif harm > tolerance and p_value < alpha:
        status = "FAIL"
    else:
        status = "WATCH"
    return {"status": status, "harm": harm, "worst_case": worst_case, "tolerance": tolerance}
