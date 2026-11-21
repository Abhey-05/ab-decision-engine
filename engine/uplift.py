"""
Uplift modelling: WHO should get the treatment?

A normal model predicts "will this user buy?". An uplift model predicts
"how much MORE likely is this user to buy IF we treat them?" That splits
users into four kinds:
    Persuadables  - buy only if treated            -> target these
    Sure things   - buy either way                  -> treating them is wasted
    Lost causes   - don't buy either way            -> treating them is wasted
    Sleeping dogs - buy only if NOT treated         -> treating them hurts

T-learner ("two models"): train one model on control users and one on
treated users. For any user, uplift = P(buy | treated model) - P(buy | control model).
S-learner ("single model"): one model with "treated" as an extra feature;
uplift = prediction with treated=1 minus prediction with treated=0.

We check the model on users it was not trained on with a Qini curve: sort users by
predicted uplift, then count how many extra purchases we'd get by treating
only the top X%. A good model rises faster than the straight "random" line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

FEATURES = ["is_new_user", "is_mobile", "city_tier", "price_band_num", "prior_sessions"]
GBM_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                  min_samples_leaf=300, subsample=0.8, random_state=0)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "is_new_user": df["is_new_user"].to_numpy(),
        "is_mobile": (df["device"] == "mobile").astype(int).to_numpy(),
        "city_tier": df["city_tier"].to_numpy(),
        "price_band_num": df["price_band"].map({"low": 0, "mid": 1, "high": 2}).to_numpy(),
        "prior_sessions": df["prior_sessions"].to_numpy(),
    })


def qini_curve(uplift: np.ndarray, treated: np.ndarray, outcome: np.ndarray, n_points: int = 101) -> pd.DataFrame:
    """Qini curve on held-out users.

    For the top-k users by predicted uplift:
        incremental(k) = buyers among treated in top k
                         - buyers among control in top k * (treated count / control count)
    i.e. extra purchases vs. what the same number of users would have done
    without treatment. The random baseline is a straight line from 0 to the
    total incremental purchases.
    """
    order = np.argsort(-uplift, kind="stable")
    t, y = treated[order], outcome[order]
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = cum_yt - np.where(cum_c > 0, cum_yc * cum_t / cum_c, 0.0)
    n = len(uplift)
    idx = np.unique(np.linspace(0, n, n_points).astype(int))
    inc_at = np.concatenate([[0.0], inc])[idx]
    frac = idx / n
    return pd.DataFrame({"fraction": frac, "model": inc_at, "random": frac * inc_at[-1]})


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model curve and the random line, divided by the
    number of points. > 0 means the model ranks users better than chance."""
    return float(np.trapezoid(curve["model"] - curve["random"], curve["fraction"]))


def _fit_predict(X_tr, t_tr, y_tr, X_te):
    """Train T- and S-learners on one half, predict on the other half."""
    m_c = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
    m_t = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
    p_c = m_c.predict_proba(X_te)[:, 1]
    p_t = m_t.predict_proba(X_te)[:, 1]

    m_s = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr.assign(treated=t_tr), y_tr)
    uplift_s = (m_s.predict_proba(X_te.assign(treated=1))[:, 1]
                - m_s.predict_proba(X_te.assign(treated=0))[:, 1])
    return p_c, p_t, uplift_s


def fit_uplift(df: pd.DataFrame, outcome: str = "purchased", seed: int = 0) -> dict:
    """Train T- and S-learners with 2-fold cross-fitting and evaluate them.

    Why cross-fitting: purchases are rare (~1.2%), so a single 30% hold-out
    has only a few hundred buyers and the Qini curve is very noisy. Instead
    we split users in half, train on half A and score half B, then swap.
    Every user gets a score from a model that never saw them, so the
    evaluation is still honest, but it uses all users.
    """
    X = _features(df)
    t = (df["variant"] == "treatment").astype(int).to_numpy()
    y = df[outcome].to_numpy()
    idx_a, idx_b = train_test_split(np.arange(len(df)), test_size=0.5, random_state=seed, stratify=t)

    p_c, p_t, up_s = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for train_idx, score_idx in [(idx_a, idx_b), (idx_b, idx_a)]:
        pc, pt, us = _fit_predict(X.iloc[train_idx], t[train_idx], y[train_idx], X.iloc[score_idx])
        p_c[score_idx], p_t[score_idx], up_s[score_idx] = pc, pt, us
    uplift_t = p_t - p_c

    curve_t = qini_curve(uplift_t, t, y)
    curve_s = qini_curve(up_s, t, y)
    scored = X.assign(treated=t, outcome=y, p_control=p_c, p_treated=p_t, uplift=uplift_t)
    return {
        "scored": scored,
        "curve_t": curve_t, "curve_s": curve_s,
        "qini_t": qini_coefficient(curve_t), "qini_s": qini_coefficient(curve_s),
        "n_users": len(df),
        # Qini counts extra purchases among the TREATED users in the top-k.
        # Multiply by this to express it per 100,000 visitors.
        "per_100k": 100_000 / t.sum(),
    }


def targeting_summary(curve: pd.DataFrame, top_fraction: float) -> dict:
    """'Treat only the top X%' vs 'treat everyone', in incremental purchases."""
    row = curve.iloc[(curve["fraction"] - top_fraction).abs().argmin()]
    everyone = float(curve["model"].iloc[-1])
    best = curve.loc[curve["model"].idxmax()]
    return {
        "top_fraction": float(row["fraction"]),
        "incremental_top": float(row["model"]),
        "incremental_random": float(row["random"]),
        "incremental_everyone": everyone,
        # Only meaningful when treating everyone produces a positive gain.
        "share_captured": float(row["model"] / everyone) if everyone > 0 else None,
        "best_fraction": float(best["fraction"]),
        "best_incremental": float(best["model"]),
    }


def user_types(scored: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    """Label each user as one of the four uplift types, using the
    model's predicted purchase probability with and without treatment.
    `threshold` = how big a predicted change counts as a real effect."""
    base_high = scored["p_control"] >= scored["p_control"].median()
    return pd.Series(np.select(
        [scored["uplift"] > threshold, scored["uplift"] < -threshold, base_high],
        ["Persuadables", "Sleeping dogs", "Sure things"], "Lost causes"), index=scored.index)


def uplift_facts(fit: dict) -> dict:
    """Small, rounded summary of the uplift model for the AI analyst's JSON."""
    curve = fit["curve_t"].copy()
    curve[["model", "random"]] *= fit["per_100k"]
    s20 = targeting_summary(curve, 0.2)
    types = user_types(fit["scored"]).value_counts(normalize=True)
    return {
        "method": "T-learner (gradient boosting), 2-fold cross-fitted",
        "extra_purchases_are_per_visitors": 100_000,
        "top_share_pct": 20,
        "qini_coefficient_t_learner": round(fit["qini_t"] * fit["per_100k"]),
        "qini_coefficient_s_learner": round(fit["qini_s"] * fit["per_100k"]),
        "extra_purchases_per_100k_visitors_if_treat_everyone": round(s20["incremental_everyone"]),
        "extra_purchases_per_100k_visitors_if_treat_top_20_pct": round(s20["incremental_top"]),
        "best_share_to_treat_pct": round(s20["best_fraction"] * 100),
        "extra_purchases_per_100k_visitors_at_best_share": round(s20["best_incremental"]),
        "user_types_pct": {k: round(v * 100, 1) for k, v in types.items()},
    }
