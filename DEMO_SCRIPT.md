# Demo script (2–3 minutes)

**Before the interview:** open the live app 5–10 minutes early. Free Streamlit apps go to sleep when idle and take ~30 s to wake. Click through every page once for each experiment so everything is cached and instant.

---

## The walkthrough

**Opening line (10 s):**
"This is an experimentation decision engine I built for an online jewellery store. It takes raw A/B test data and answers the questions a PM actually has: can I trust this test, did it work, for whom, did it break anything, and what should we do next?"

### 1. Home: Price Breakup Card (20 s)
Sidebar is already on **Price Breakup Card**.
> "The idea: show customers how the price is built up, with gold value, making charges, stones and tax. The engine says **Ship to all**: purchases went from 1.16% to 1.33%, about +15%, and the confidence range is entirely above zero."

Point at the 8-step pipeline: "Every experiment goes through the same eight steps, and the first step after data is a trust check."

### 2. Results (30 s)
> "These tiles are the headline: **+15.0% lift, 95% CI +4.2% to +25.9%, p = 0.0065**, so it's very unlikely to be luck. The forest plot shows every metric with its uncertainty: add-to-cart is up too, and the guardrails (order value, returns, page load) show no significant harm. Order value and returns are marked 'Watch' because they're too noisy to *prove* safe, so I'd monitor them after launch."

### 3. Switch to Video Reviews: Trust checks (25 s)
Pick **Video Reviews (broken)** in the sidebar, open **Trust checks**.
> "This one *looks* like a significant +12% win. But we planned 50/50 and got 61,000 vs 59,000. With fair coin flips that would happen less than once in a million tests. Slow-loading treatment users were never logged, so the groups aren't comparable. The engine **refuses to give a recommendation**. In the synthetic data I know the true effect is +5%, so the bug more than doubled the apparent lift."

### 4. Switch to Try-at-Home: Segments → Uplift (40 s)
Pick **Try-at-Home Button**, open **Segments**.
> "On average this is flat, so a naive reading says don't ship. But high-price-band shoppers are up 73% and tier-1 cities 39%, while tier-2/3 cities go down. The two cancel out."

Open **Uplift targeting**, drag the slider to ~30%.
> "An ML uplift model predicts each user's personal benefit. Treating only the top 30% gives about **2.9× the extra purchases** of treating everyone, and the curve beats random targeting. So the recommendation is a **targeted rollout**."

### 5. Switch to 10% Off Pop-up: AI Analyst (30 s)
Pick **10% Off Pop-up**, open **AI Analyst**, click **Generate readout**.
> "Purchases are up 20%, but order value dropped 21% and returns rose 43%, so two guardrails failed. The effect also fades from +41% in the first four days to +11% in week two, a novelty effect. The engine says **Iterate**. The AI writes the readout, but only from this JSON fact sheet, and every number it writes is checked: here, all verified."

### Close (10 s)
> "The statistics are all standard, tested methods, verified against statsmodels and an A/A simulation. The decision comes from explicit rules, and the AI only explains. As a PM, that's what I'd want: fast, consistent and auditable experiment reads."

---

## Likely questions, short answers

**Why is the data synthetic?**
Real experiment data is confidential. Synthetic data with *planted* effects lets me prove the methods work: I planted +15% and the engine measured +15.0%, with the true value inside the confidence interval.

**What's a p-value, in one line?**
If the change did nothing, how often would luck alone give a difference this big. Below 5% we call it significant.

**What's a confidence interval?**
Like a weather forecast range: we're 95% confident the true effect is between +4% and +26%. If the whole range is above zero, it helped.

**What is SRM and why does it matter?**
Sample ratio mismatch: the groups aren't the size we planned. It means users are being lost or duplicated in one group, usually not at random, so any result is biased. It's one of the most common ways real tests silently break.

**Why did product view rate drop in Price Breakup?**
The generator plants no effect on views, so that's a genuine false positive. Test enough metrics and some will look significant by chance. That's why the decision rests on one pre-chosen primary metric, and secondary metrics get a multiple-testing correction.

**What is Benjamini-Hochberg?**
A correction for testing many things at once. With 13 segments, about one will look special by pure luck. It raises the bar so we don't chase noise.

**Why guardrails?**
A discount can raise purchases while destroying margin and increasing returns. Guardrails make "don't break the business" explicit, with agreed tolerances (AOV −3%, returns +5%, page load +5%).

**What's an uplift model vs a normal model?**
A normal model predicts "will they buy". An uplift model predicts "how much *more* likely are they to buy *because* we showed them this". That separates persuadables from people who'd buy anyway, or who are put off ("sleeping dogs").

**T-learner vs S-learner?**
T-learner trains two models (one on control, one on treatment) and subtracts. S-learner trains one model with "treated" as a feature. I show both; T-learner is usually better when the effect differs a lot between groups.

**How do you know the uplift model works?**
Qini curve on users the model never saw (2-fold cross-fitting). If targeting by the model beats random targeting, it's learned something real.

**What's a novelty effect?**
Users react to something because it's new, then get used to it. The pop-up's lift dropped from +41% to +11%, so the long-run effect is smaller than the headline.

**How do you stop the AI hallucinating?**
It never computes anything. It only gets a JSON of pre-computed results, follows a fixed template, and code extracts every number it writes and checks it against the JSON. Without an API key, a template writes the same readout.

**How many users do you need?**
Use the *Plan a test* page. For a 1.2% purchase rate and a +15% lift, about 62,000 per group (80% power). Rare purchases need big samples, which is why add-to-cart is a useful leading indicator.

**What would you add next?**
CUPED (using pre-experiment behaviour to reduce noise and run shorter tests), sequential testing so teams can peek safely, and connecting it to a real event pipeline.

**How would this help a jewellery business specifically?**
Purchases are rare and high-value, so tests are slow and noisy, and margin and returns matter a lot. Trust checks, guardrails on AOV and returns, and targeting high-intent segments (e.g. try-at-home for expensive pieces in tier-1 cities) are exactly the levers that matter.
