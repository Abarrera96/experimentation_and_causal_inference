# Causal Inference: From A/B Tests to Observational Methods
## Three notebooks, two real datasets, one causal hierarchy

---

## The Problem

How do you measure the causal effect of a change when you cannot always run a perfect experiment?

This repository answers that question by applying three causal inference methods to two real datasets from well-known companies. Each notebook tackles the same core challenge — isolating cause from correlation — using a different method, and explains why that method is the right tool for that specific situation.

---

## Project Structure

```
causal-inference/
│
├── data/
│   ├── cookie_cats.csv           ← A/B test dataset (Tactile Entertainment)
│   ├── card_krueger_wide.csv     ← Double ML, one row per restaurant
│   └── card_krueger_long.csv     ← DiD regression, one row per restaurant per period
│
├── notebooks/
│   ├── 01_ab_test.ipynb          ← Cookie Cats — real randomized experiment
│   ├── 02_diff_in_diff.ipynb     ← Card & Krueger — natural experiment
│   └── 03_double_ml_dowhy.ipynb  ← Card & Krueger — observational causal inference
│
├── download_data.py              ← downloads all datasets automatically
├── requirements.txt
└── README.md
```

---

## Datasets

### Dataset 1 — Cookie Cats (Notebook 01)
**Company:** Tactile Entertainment
**Product:** Cookie Cats — one of the most downloaded mobile puzzle games ever

Cookie Cats features "gates" that force players to wait or make an in-app purchase to keep playing. Tactile Entertainment ran a real A/B test on 90,189 players to decide whether moving the gate from level 30 to level 40 affected player retention.

**Why it's a real A/B test:** players were randomly assigned to `gate_30` (control) or `gate_40` (treatment) at install time. True randomization — the gold standard of causal inference.

| Variable | Description |
|----------|-------------|
| `userid` | Unique player ID |
| `version` | `gate_30` (control) or `gate_40` (treatment) |
| `sum_gamerounds` | Rounds played in first 14 days |
| `retention_1` | Did the player return 1 day after install? (binary) |
| `retention_7` | Did the player return 7 days after install? (binary) |

**File:** `data/cookie_cats.csv` — 90,189 players

---

### Dataset 2 — Card & Krueger 1994 (Notebooks 02 + 03)
**Study:** Minimum Wages and Employment (American Economic Review, 1994)
**Authors:** David Card (Nobel Prize 2021) & Alan Krueger

In April 1992, New Jersey raised its minimum wage from $4.25 to $5.05. Pennsylvania did not. Card & Krueger surveyed 410 fast-food restaurants in both states before (February 1992) and after (November 1992) the change.

This is not a randomized experiment — states chose their own policies. That is exactly why we need DiD and Double ML instead of a simple t-test.

| Variable | Description |
|----------|-------------|
| `nj` | 1 if New Jersey (treated), 0 if Pennsylvania (control) |
| `fte` | Full-time equivalent employees — pre-period |
| `fte2` | Full-time equivalent employees — post-period |
| `fte_change` | `fte2 - fte` — change in employment (outcome for Double ML) |
| `wage_st` | Starting wage — pre-period |
| `wage_st2` | Starting wage — post-period |
| `wage_change` | `wage_st2 - wage_st` — change in minimum wage (treatment for Double ML) |
| `bk`, `kfc`, `roys`, `wendys` | Chain dummies |
| `co_owned` | 1 if company-owned (not franchise) |

**Two files, two formats, two different reasons:**

| File | Format | Used in | Why |
|------|--------|---------|-----|
| `card_krueger_long.csv` | Two rows per restaurant (pre + post) | `02_diff_in_diff.ipynb` | DiD regression needs a `post` column: `fte_obs ~ nj + post + nj:post` |
| `card_krueger_wide.csv` | One row per restaurant | `03_double_ml_dowhy.ipynb` | Double ML needs `fte_change` and `wage_change` as single columns — one row per unit |

---

## The Three Notebooks

### 01 — A/B Test (`01_ab_test.ipynb`)
**Dataset:** Cookie Cats · 90,189 players
**Question:** Does moving the gate from level 30 to level 40 reduce Day-7 player retention?

The gold standard of causal inference. Players were randomly assigned at install time — randomization guarantees causality by design. No assumptions about confounders needed.

**What the notebook covers:**
- SRM check — chi-squared test on group sizes, imbalance fixed by downsampling
- Sample size calculation: MDE = 1pp, α = 0.05, power = 80% → required n = 24,178
- Z-test for proportions (binary metric + n > 44,000 → CLT applies)
- Guardrail metric: `sum_gamerounds` (Welch t-test)
- CUPED variance reduction: ρ = 0.28, 7.8% variance reduction
- Common mistakes: HARKing, peeking, SRM, SUTVA, multiple comparisons

**Result:** gate_30 significantly retains more players at Day 7.
p = 0.003 · absolute difference = +0.78pp · CI [0.27pp, 1.29pp] · guardrail OK ✓

---

### 02 — Difference-in-Differences (`02_diff_in_diff.ipynb`)
**Dataset:** Card & Krueger long format · 410 restaurants · 2 periods
**Question:** Did raising the minimum wage in New Jersey reduce employment?

When randomization is impossible, we exploit natural experiments. NJ raised the minimum wage; PA did not. By comparing how employment *changed* in NJ vs PA, we cancel fixed differences between states and common time trends.

**What the notebook covers:**
- Why a simple t-test would be wrong (different baseline levels)
- Parallel trends verification — the critical assumption
- Manual ATT calculation: replicating Table 3 from the original paper
- DiD regression: `fte_obs ~ nj + post + nj:post` (OLS — continuous outcome)
- Standard errors: only 2 clusters → OLS without correction, limitation declared
- Why DiD and not a t-test

**The logic:**
```
ATT = (NJ_post − NJ_pre) − (PA_post − PA_pre)
    = +0.59 − (−2.17) = +2.75 FTE
```

**Result:** β₃ = +2.75 FTE · p = 0.103 · CI [−0.56, 6.07]
No evidence of job destruction. Classical economics prediction refuted.
Exact replication of Card & Krueger (1994) Table 3 ✓

---

### 03 — Double ML + DoWhy (`03_double_ml_dowhy.ipynb`)
**Dataset:** Card & Krueger wide format · 351 restaurants (after dropping NAs)
**Question:** By how much does a $1 increase in the minimum wage change employment, controlling for chain type, ownership and region?

What if we only had observational data — no natural experiment? This notebook uses the professional causal inference workflow adopted by Amazon, Microsoft, Uber, and Airbnb: **DoWhy + Double ML**.

DoWhy forces explicit causal assumptions via a DAG before estimating anything. Double ML removes non-linear confounder effects using ML, then estimates the causal effect on the residuals. Four refutation tests stress-test the result.

**Treatment (T):** `wage_change` — change in minimum wage per restaurant ($)
**Outcome (Y):** `fte_change` — change in FTE employees
**Confounders (X):** chain type (bk, kfc, roys, wendys) · ownership (co_owned) · region (southj, centralj, northj, pa1, pa2)

**The DoWhy workflow:**
1. **Model** — define the DAG in DOT format → `CausalModel`
2. **Identify** — backdoor criterion → unconfoundedness assumption declared
3. **Estimate** — Double ML via econml · GradientBoosting for ml_g and ml_t · StatsModelsLinearRegression for model_final (continuous outcome)
4. **Refute** — 4 robustness tests

**Refutation results (all pass ✓):**

| Test | Original θ | New θ | p-value | Pass? |
|------|-----------|-------|---------|-------|
| Placebo treatment | +1.27 | +0.08 | 0.98 | ✓ |
| Random common cause | +1.27 | +1.71 | 0.64 | ✓ |
| Data subset | +1.27 | +1.38 | 0.84 | ✓ |
| Bootstrap | +1.27 | +1.96 | 0.74 | ✓ |

**Result:** θ = +1.27 FTE · p = 0.194 · CI [−1.59, 4.12]
Consistent with DiD. No evidence of job destruction.

---

## Results Summary

| Notebook | Method | Dataset | Causal effect | p-value | Significant |
|----------|--------|---------|--------------|---------|-------------|
| 01 | A/B Test | Cookie Cats | −0.78pp D7 retention | 0.003 | ✓ Yes |
| 02 | DiD | Card & Krueger long | +2.75 FTE | 0.103 | ✗ No |
| 03 | Double ML | Card & Krueger wide | +1.27 FTE | 0.194 | ✗ No |

**Key finding across notebooks 02 and 03:** both methods point in the same direction — positive effect, no evidence of job destruction. The classical economics prediction (negative effect) is not supported by the data.

---

## The Causal Hierarchy

```
Level 1 — A/B Test (RCT)             Notebook 01
  Causality guaranteed by design.
  Most credible. No assumptions about confounders.

Level 2 — Quasi-experiment (DiD)     Notebook 02
  Near-random variation from a natural event.
  One testable assumption: parallel trends.

Level 3 — Observational (Double ML)  Notebook 03
  Only historical data. Most assumptions.
  Validity rests on conditional ignorability.
  Defended via DAG + 4 DoWhy refutation tests.
```

Every method below the top is an approximation of what a perfect experiment would give us. The further down, the more assumptions we must defend.

---

## Why This Portfolio Matters

Most DS portfolios show predictive models. This project demonstrates something harder: **causal reasoning**.

The skills shown here:
- Designing experiments and understanding their limitations (MDE, SRM, CUPED)
- Recognizing when randomization is impossible and choosing the right alternative
- Applying DiD when a natural event creates near-random variation
- Using Double ML + DoWhy when only observational data is available
- Stress-testing causal estimates before presenting conclusions
- Communicating the hierarchy of evidence clearly

These are the skills that separate a DS who builds models from one who drives business decisions.

---

## Setup

```bash
git clone https://github.com/Abarrera96/experimentation_and_causal_inference
cd experimentation_and_causal_inference
pip install -r requirements.txt
python download_data.py
```

Run notebooks in order: `01` → `02` → `03`

---

## Stack

```
Python 3.11     pandas==2.2.3       numpy              matplotlib==3.9.2
scipy==1.14.1   statsmodels==0.14.4 scikit-learn==1.6.1 xgboost==2.1.3
lightgbm==4.5.0 econml==0.16.0      dowhy==0.14         linearmodels==6.1
```

---

## References

Card, D., & Krueger, A. B. (1994). Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772–793.

Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1), C1–C68.

Cookie Cats A/B Test Dataset. Tactile Entertainment / DataCamp.

---

## Author

Data Scientist with 5 years of experience.
Passionate about causal inference and evidence-based decision making.

[LinkedIn](https://www.linkedin.com/in/antonio-barrera-reyzabal) · [GitHub](https://github.com/Abarrera96)