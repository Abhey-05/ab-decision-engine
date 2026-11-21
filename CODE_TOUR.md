# Code tour

A beginner-friendly map of the code, so you can answer "how did you build this?".

## The big idea

The project is split in two:

- **`engine/`**: all the maths and logic, in plain Python. It has **no Streamlit code**, so it can be tested on its own.
- **`app.py`, `ui.py`, `app_pages/`**: the website. These files only *display* what the engine computed.

Data flows one way: **generate data → analyse → display**. The AI analyst sits at the very end and only explains.

## What each file does

| File | In one sentence |
|---|---|
| `engine/data.py` | Creates the four fake-but-realistic experiments, with known effects planted in them. |
| `engine/stats.py` | The statistics toolbox: z-test, Welch's t-test, bootstrap, sample size, Benjamini-Hochberg, guardrail check. |
| `engine/checks.py` | Trust checks: is the 50/50 split fair (SRM), any duplicate users, are the groups balanced. |
| `engine/metrics.py` | Applies the toolbox to every metric: primary (purchases), secondary, guardrails, and daily numbers. |
| `engine/funnel.py` | Counts users at each step: viewed → added to cart → purchased. |
| `engine/segments.py` | Repeats the purchase comparison inside each user group, with a multiple-testing correction. |
| `engine/uplift.py` | Machine learning: predicts how much each user is helped by the change; builds the Qini curve. |
| `engine/anomalies.py` | Simple rules that flag problems: SRM, guardrail breaches, odd segments, fading effects, daily spikes. |
| `engine/decision.py` | Turns everything into one of five decisions, with reasons, using explicit if/else rules. |
| `engine/pipeline.py` | Runs all of the above in order with one call: `analyze(df)`. |
| `engine/analyst.py` | Builds the JSON fact sheet, calls Gemini (or the template), and checks every number. |
| `ui.py` | Shared website helpers: caching, the sidebar picker, colours, the recommendation card. |
| `app.py` | The entry point: sets up the page, the sidebar and the list of pages. |
| `app_pages/*.py` | One file per page (Home, Trust checks, Results, Segments, Uplift, AI Analyst, Plan, How it works). Not called `pages/` on purpose: that name makes Streamlit auto-discover pages and bypass `app.py` on a fresh server. |
| `tests/test_engine.py` | Proves the statistics and decisions are right. |
| `tests/test_app.py` | Opens every page for every experiment and checks nothing crashes. |

## The 10 most important functions

**1. `simulate(key)`** in `engine/data.py`
Makes one experiment. For each user it draws attributes (new or returning, mobile or desktop, city tier, price band), works out their chance of adding to cart and buying, multiplies in the planted effect if they're in treatment, then "flips weighted coins" to decide what they did. Fixed seed = same data every time. It also returns the *true* effect so tests can check the engine finds it.

**2. `two_proportion_ztest(...)`** in `engine/stats.py`
The core test. Takes "x buyers out of n users" for each group and returns the lift, a 95% confidence interval and a p-value. Plain English: "is the gap between 1.16% and 1.33% bigger than coin-toss luck would produce?" Tested against `statsmodels`.

**3. `bootstrap_ci(...)`** in `engine/stats.py`
For order values, which are very uneven (a few ₹1-lakh orders among many ₹10k ones). It re-draws the orders randomly 2,000 times and sees how much the average wobbles. The middle 95% of the wobble is the confidence interval. It's "vectorised", meaning all 2,000 re-draws happen in one NumPy operation instead of a slow loop.

**4. `srm_check(...)`** in `engine/checks.py`
Chi-square test on the group sizes. If 120,000 users were split by fair coin flips, 61,000 vs 59,000 is essentially impossible (p ≈ 0.00000001), so something in the pipeline is broken.

**5. `guardrail_check(...)`** in `engine/stats.py`
Decides Pass / Watch / Fail for a guardrail. **Pass** if even the worst end of the confidence interval is within tolerance. **Fail** if it got worse than tolerance *and* that's statistically significant. **Watch** otherwise (too noisy to tell).

**6. `benjamini_hochberg(...)`** in `engine/stats.py`
When you test many things, some look good by luck. This sorts the p-values and raises the bar for each one, so the share of false discoveries stays around 5%. Used for secondary metrics and segments.

**7. `segment_lifts(df)`** in `engine/segments.py`
Loops over user groups (new/returning, device, city tier, price band, visit history), runs the z-test inside each, applies Benjamini-Hochberg, and marks groups that move opposite to the overall result.

**8. `fit_uplift(df)`** in `engine/uplift.py`
Trains gradient-boosting models. **T-learner:** one model learns "chance of buying without the change", another "with the change"; the difference is each user's predicted uplift. **S-learner:** one model with "treated" as an input. Users are split in half, so each half is scored by a model trained on the *other* half (cross-fitting), which keeps the evaluation honest.

**9. `qini_curve(...)`** in `engine/uplift.py`
Sorts users from "most helped" to "least helped" and counts extra purchases as you treat more of them. If the curve rises faster than a straight line (random targeting), the model works. The peak tells you the best share of users to target.

**10. `decide(...)`** in `engine/decision.py`
The rulebook, deliberately simple if/else logic so anyone can audit it:
broken test → *No decision*; purchases up but a guardrail failed → *Iterate*; purchases up and nothing hurt → *Ship to all*; flat but some group clearly helped → *Targeted rollout*; otherwise → *Don't ship*. Each branch writes plain-English reasons.

**Bonus: `verify_numbers(text, facts)`** in `engine/analyst.py`
Pulls every number out of the AI's text with a regular expression and checks it exists in the JSON fact sheet (allowing rounding, so "15%" matches 15.04). Anything that doesn't match is shown as a warning. This is the anti-hallucination check.

## How the website stays fast

- `@st.cache_data` in `ui.py` means data generation (~0.3 s per experiment), analysis and uplift training (~5 s) run **once** and are then reused.
- The Home page quietly trains the uplift model for the selected experiment, so the Uplift and AI pages open instantly.

## How to change things

- **Change an effect size:** edit `_effects()` in `engine/data.py`, then run `pytest` to see which conclusions change.
- **Change guardrail tolerances:** `DEFAULT_GUARDRAILS` in `engine/stats.py`.
- **Change the AI model:** `MODEL` in `engine/analyst.py`.
- **Add a page:** create `app_pages/new_page.py` and add one `st.Page(...)` line in `app.py`.
