"""
Synthetic jewellery e-commerce A/B test data.

Why synthetic? Real experiment data is private, and synthetic data lets us
*plant* known effects (e.g. "+15% purchases") and then check that the engine
finds them. If the engine recovers the truth we planted, we can trust it.

How it works (per user):
    1. Draw user attributes (new vs returning, device, city tier, ...).
    2. Compute baseline probabilities for each funnel step:
           view product -> add to cart -> purchase
       Desktop, returning and tier-1 users convert a bit better; slow pages
       convert worse; expensive products convert less often.
    3. Rescale those probabilities so the control group hits round, realistic
       targets (8% add-to-cart, 1.2% purchase).
    4. Multiply in the treatment effect for treated users. Each experiment has
       its own effect pattern (see EXPERIMENTS below).
    5. Flip weighted coins to decide what each user actually did.

Everything uses a fixed random seed, so the app shows the same numbers every
time it starts. Seeds were picked so that random noise doesn't hide the
planted story (with ~700 purchases per arm, a true +15% lift is only
detected ~70% of the time - see the "Plan a test" page).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_USERS = 120_000          # users per experiment
TARGET_ATC = 0.080         # control add-to-cart rate (per user)
TARGET_PURCHASE = 0.012    # control purchase rate (per user)
BASE_RETURN_RATE = 0.07    # share of orders that get returned

# Human-readable description of each experiment. Keys are used everywhere
# in the code; "name" is what the user sees.
EXPERIMENTS = {
    "price_breakup": {
        "name": "Price Breakup Card",
        "change": "Show a price breakup (gold value, making charges, stones, tax) on the product page",
        "story": "Clear winner",
        "seed": 2047,
    },
    "try_at_home": {
        "name": "Try-at-Home Button",
        "change": "Add a 'Book a free try-at-home' button on the product page",
        "story": "Flat on average, but a hidden winner in some segments",
        "seed": 2027,
    },
    "discount_popup": {
        "name": "10% Off Pop-up",
        "change": "Show a '10% off your first order' pop-up after 20 seconds",
        "story": "Wins on purchases but breaks guardrails (order value, returns)",
        "seed": 2028,
    },
    "video_reviews": {
        "name": "Video Reviews (broken)",
        "change": "Add customer video reviews to the product page",
        "story": "Untrustworthy test: sample ratio mismatch",
        "seed": 2033,
    },
}

# Median order value (rupees) by the price band of the product being browsed.
AOV_MEDIAN = {"low": 9_000, "mid": 24_000, "high": 65_000}


# ---------------------------------------------------------------------------
# Step 1 + 2: users and their baseline (control) probabilities
# ---------------------------------------------------------------------------

def _make_users(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw user attributes. Proportions are loosely realistic for Indian e-commerce."""
    is_new = rng.random(n) < 0.55
    device = np.where(rng.random(n) < 0.70, "mobile", "desktop")
    city_tier = rng.choice([1, 2, 3], size=n, p=[0.40, 0.35, 0.25])
    price_band = rng.choice(["low", "mid", "high"], size=n, p=[0.45, 0.35, 0.20])
    # New users have no prior sessions; returning users have 1+ (long tail).
    prior_sessions = np.where(is_new, 0, np.minimum(rng.geometric(0.15, n), 60))
    exposure_day = rng.integers(1, 15, size=n)  # 1..14

    # Page load: mobile and tier-3 cities are slower. Log-normal = long right tail.
    median_load = np.where(device == "mobile", 2100, 1400) * np.where(city_tier == 3, 1.15, 1.0)
    page_load_ms = median_load * rng.lognormal(0.0, 0.35, n)

    return pd.DataFrame({
        "is_new_user": is_new.astype(int),
        "device": device,
        "city_tier": city_tier,
        "price_band": price_band,
        "prior_sessions": prior_sessions.astype(int),
        "exposure_day": exposure_day,
        "page_load_ms": page_load_ms,
    })


def _baseline_probs(u: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_view, p_cart_given_view, p_buy_given_cart) before any treatment."""
    new = u["is_new_user"].to_numpy() == 1
    desktop = u["device"].to_numpy() == "desktop"
    tier = u["city_tier"].to_numpy()
    band = u["price_band"].to_numpy()
    prior = u["prior_sessions"].to_numpy()
    load = u["page_load_ms"].to_numpy()

    p_view = np.where(new, 0.56, 0.65)

    cart = np.ones(len(u))
    cart *= np.where(desktop, 1.15, 0.94)
    cart *= np.where(new, 0.85, 1.20)
    cart *= np.select([band == "low", band == "mid"], [1.10, 1.00], 0.80)
    cart *= np.select([tier == 1, tier == 2], [1.10, 1.00], 0.85)
    cart *= 1 + 0.01 * np.minimum(prior, 20)
    cart *= (1800 / load) ** 0.4          # slow pages lose shoppers

    buy = np.ones(len(u))
    buy *= np.where(desktop, 1.10, 1.00)
    buy *= np.where(new, 1.00, 1.15)
    buy *= np.select([band == "low", band == "mid"], [1.15, 1.00], 0.75)

    # Step 3: rescale so the population hits the target rates exactly.
    cart *= TARGET_ATC / np.mean(p_view * cart)
    buy *= TARGET_PURCHASE / np.mean(p_view * cart * buy)
    return p_view, cart, buy


# ---------------------------------------------------------------------------
# Step 4: treatment effects, one function per experiment
# Each returns multipliers applied to treated users only:
#   cart_mult  - multiplies P(add to cart | view)
#   buy_mult   - multiplies P(purchase | cart)
#   aov_mult   - multiplies order value
#   ret_mult   - multiplies P(return | purchase)
#   load_mult  - multiplies page load time
# ---------------------------------------------------------------------------

def _solve_scale(base: np.ndarray, weight: np.ndarray, target_total: float) -> float:
    """Find k so that mean(base * (1 + k * weight)) == target_total.

    This is how we "calibrate" a planted effect: we choose the *shape* of the
    effect across segments, then solve for the size that gives the exact
    population-level lift we want. It is a simple linear equation.
    """
    return (target_total - base.mean()) / (base * weight).mean()


def _effects(key: str, u: pd.DataFrame, pv, pc, pb) -> dict:
    n = len(u)
    new = u["is_new_user"].to_numpy() == 1
    tier = u["city_tier"].to_numpy()
    band = u["price_band"].to_numpy()
    prior = u["prior_sessions"].to_numpy()
    day = u["exposure_day"].to_numpy()
    ones = np.ones(n)
    eff = {"cart": ones.copy(), "buy": ones.copy(), "aov": 1.0, "ret": 1.0, "load": 1.0}

    if key == "price_breakup":
        # Transparency helps most where trust is lowest: new users and
        # expensive pieces. Weight = how strongly each user responds.
        w = 0.25 + 1.0 * new + 0.9 * (band == "high")
        a = _solve_scale(pv * pc, w, TARGET_ATC * 1.1125)              # ATC 8.0% -> 8.9%
        eff["cart"] = 1 + a * w
        b = _solve_scale(pv * pc * eff["cart"] * pb, w, TARGET_PURCHASE * 1.15)  # purchase +15%
        eff["buy"] = 1 + b * w
        eff["aov"] = 0.92      # pieces bought skew pricier, but people trade down a little: net AOV ~ -2%

    elif key == "try_at_home":
        # Big effect for tier-1 cities buying expensive pieces (try-at-home
        # is only valuable there); slightly negative elsewhere (distraction).
        t1, high = tier == 1, band == "high"
        positive = np.select([t1 & high, t1 | high], [0.90, 0.25], 0.0)
        negative_zone = ~(t1 | high)
        # Choose the negative effect so the *overall* purchase lift is zero.
        base = pv * pc * pb
        k = (base * positive).sum() / (base * negative_zone).sum()
        eff["buy"] = 1 + positive - k * negative_zone

    elif key == "discount_popup":
        # Discounts pull in new / light users, annoy loyal heavy users
        # ("sleeping dogs"), and wear off over time (novelty effect).
        shape = np.select([prior <= 3, prior <= 9], [0.35, 0.10], -0.55)
        novelty = np.select([day <= 4, day <= 7], [2.5, 1.0], 0.25)
        w = shape * novelty
        eff["cart"] = ones * 1.05
        base = pv * pc * eff["cart"] * pb
        k = _solve_scale(base, w, TARGET_PURCHASE * 1.20)               # purchase +20%
        eff["buy"] = 1 + k * w
        eff["aov"] = 0.85      # 10% off + people pick cheaper pieces
        eff["ret"] = 1.6       # impulse buys get returned more
        eff["load"] = 1.02     # pop-up script, small and within tolerance

    elif key == "video_reviews":
        eff["cart"] = ones * 1.15  # modest true effect (partly cancelled by slower pages)
        eff["load"] = 1.25         # videos make pages heavier

    return eff


# ---------------------------------------------------------------------------
# Step 5: simulate behaviour
# ---------------------------------------------------------------------------

def simulate(key: str, n: int = N_USERS, seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Generate one experiment. Returns (data, truth).

    `truth` holds the *expected* control/treatment rates implied by the
    probabilities we planted, so tests can check the engine recovers them.
    """
    if key not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {key!r}")
    rng = np.random.default_rng(EXPERIMENTS[key]["seed"] if seed is None else seed)

    if key == "video_reviews":
        # Broken test: assign exactly 61,000 per arm, then (below) lose the
        # 2,000 slowest-loading treatment users, so we observe 61k vs 59k.
        n_assigned = int(round(n * 61 / 120)) * 2
        u = _make_users(rng, n_assigned)
        treated = rng.permutation(np.repeat([0, 1], n_assigned // 2))
    else:
        u = _make_users(rng, n)
        treated = (rng.random(n) < 0.5).astype(int)   # fair coin per user

    pv, pc, pb = _baseline_probs(u)
    eff = _effects(key, u, pv, pc, pb)

    t = treated == 1
    # The treatment can change page load, which in turn changes cart propensity.
    load = u["page_load_ms"].to_numpy() * np.where(t, eff["load"], 1.0)
    pc_obs = pc * (u["page_load_ms"].to_numpy() / load) ** 0.4

    p_cart = np.clip(pc_obs * np.where(t, eff["cart"], 1.0), 0, 0.95)
    p_buy = np.clip(pb * np.where(t, eff["buy"], 1.0), 0, 0.95)

    # Try-at-home: a festive sale on day 10 lifts purchases in *both* arms.
    # This gives the anomaly detector a daily spike to find.
    if key == "try_at_home":
        p_buy = np.clip(p_buy * np.where(u["exposure_day"].to_numpy() == 10, 1.8, 1.0), 0, 0.95)

    viewed = rng.random(len(u)) < pv
    carted = viewed & (rng.random(len(u)) < p_cart)
    purchased = carted & (rng.random(len(u)) < p_buy)

    band = u["price_band"].to_numpy()
    median_value = pd.Series(band).map(AOV_MEDIAN).to_numpy()
    value = median_value * rng.lognormal(0.0, 0.35, len(u)) * np.where(t, eff["aov"], 1.0)
    order_value = np.where(purchased, np.round(value, 0), 0.0)

    p_return = BASE_RETURN_RATE * np.where(band == "high", 1.3, 1.0) * np.where(t, eff["ret"], 1.0)
    returned = purchased & (rng.random(len(u)) < p_return)

    df = u.copy()
    df["page_load_ms"] = np.round(load, 0)
    df.insert(0, "variant", np.where(t, "treatment", "control"))
    df.insert(0, "experiment", key)
    df["viewed_product"] = viewed.astype(int)
    df["added_to_cart"] = carted.astype(int)
    df["purchased"] = purchased.astype(int)
    df["order_value"] = order_value
    df["returned"] = returned.astype(int)

    # Expected (true) rates, used only by tests.
    p_purchase = pv * p_cart * p_buy
    truth = {
        "purchase_control": float(p_purchase[~t].mean()),
        "purchase_treatment": float(p_purchase[t].mean()),
    }

    if key == "video_reviews":
        # The bug: treatment users whose (heavier) page took longest to load
        # never fired the "exposure" event, so they vanish from the data.
        slow_treated = df.index[t][np.argsort(load[t])[-2_000:]]
        df = df.drop(index=slow_treated)

    df = df.reset_index(drop=True)
    df.insert(0, "user_id", [f"{key[:2].upper()}{i:06d}" for i in range(len(df))])
    cols = ["user_id", "experiment", "variant", "is_new_user", "device", "city_tier",
            "price_band", "prior_sessions", "viewed_product", "added_to_cart", "purchased",
            "order_value", "returned", "page_load_ms", "exposure_day"]
    return df[cols], truth


def generate_experiment(key: str) -> pd.DataFrame:
    """Public entry point used by the app: the data only, no truth."""
    return simulate(key)[0]
