# THEORY: trading a predictable return when trading is costly

This file derives everything the code in `s2p/lq.py`, `s2p/dp.py` and `s2p/backtest.py` uses.
It is written to be explained aloud: every step is one idea.

**Source note.** The solution below is derived from scratch. The paper (Garleanu & Pedersen 2013,
*Dynamic Trading with Predictable Returns and Transaction Costs*, Journal of Finance 68(6)) was
**not** fetched in this project, so nothing here has been checked against its text. Section 2.5
checks our solution against the closed form for the special case Λ = λΣ *as I recall it from the
paper*. The check passes numerically, but treat equation numbers and the exact statement as
unverified until you compare with the paper yourself.

---

## 1. The problem

Time is discrete (here: one period = one hour). There are n assets and K signals.

| symbol | meaning |
|---|---|
| x_t | positions (USD) held from t to t+1, chosen at t |
| Δx_t = x_t − x_{t−1} | the trade made at t |
| f_t | K signals observed at t |
| B f_t | expected return of the n assets over (t, t+1], given the signals |
| Σ | covariance of returns |
| Λ | quadratic trading-cost matrix: trading Δx costs ½ Δx'ΛΔx |
| γ | risk aversion |
| β | per-period discount factor (0.999 per hour here) |

**Signals decay.** Each signal mean-reverts (an AR(1)): f_{t+1} = (I − Φ) f_t + ε_{t+1}. If Φ is
diagonal with entries φ_k, signal k keeps a fraction ρ_k = 1 − φ_k of its value each period. So
E_t f_{t+h} = (I − Φ)^h f_t, and its half-life is ln 0.5 / ln ρ_k periods.

**Objective.** Choose trades to maximise

    E Σ_t β^t [ x_t'B f_t − (γ/2) x_t'Σ x_t − ½ Δx_t'Λ Δx_t ]

That is, expected return minus a risk penalty minus trading costs, every period.

Two easy policies frame the problem:
* **Markowitz (naive):** ignore costs and hold x_t = (γΣ)^{−1} B f_t every period. This is
  optimal when Λ = 0, but it trades the full signal change every period.
* **Don't trade:** zero cost, zero alpha.

The optimal policy sits in between. Two questions remain: how fast to trade, and towards what target.

## 2. Solution by dynamic programming

### 2.1 The value function is quadratic

The state at t is what we hold (x_{t−1}) and what we know (f_t). Everything is linear-quadratic,
so we guess that the value of entering period t in state (x, f) is a quadratic:

    V(x, f) = −½ x'A_xx x + x'A_xf f + ½ f'A_ff f + const

Then we check the guess reproduces itself under the Bellman equation.

### 2.2 One step of the Bellman equation

    V(x_{t−1}, f_t) = max over x of  { x'B f − (γ/2) x'Σx − ½ (x − x_{t−1})'Λ(x − x_{t−1}) + β E_t V(x, f_{t+1}) }

The expectation is easy because f_{t+1} = (I−Φ)f_t + ε has mean (I−Φ)f_t and the x-terms of
V are at most linear in f:

    E_t V(x, f_{t+1}) = −½ x'A_xx x + x'A_xf (I−Φ) f + (terms without x)

Collect everything that involves x:

    −½ x'J x + x'(Λ x_{t−1} + D f),   with  J = γΣ + Λ + βA_xx,   D = B + βA_xf(I−Φ)

**First-order condition** (set the derivative in x to zero):

    x_t = J^{−1}Λ x_{t−1} + J^{−1} D f_t                                   (1)

### 2.3 The optimal rule: partial trading towards an aim portfolio

Write W = γΣ + βA_xx, so J = W + Λ. Rearranging (1):

    x_t = x_{t−1} + M (aim_t − x_{t−1}),   M = J^{−1}W,   aim_t = W^{−1} D f_t    (2)

Read it as: **each period, trade a fraction M of the way from where you are to the aim
portfolio.** Check: I − M = J^{−1}(J − W) = J^{−1}Λ, which matches the weight on x_{t−1} in (1).

### 2.4 The Riccati equations

Substitute the optimal x back into the bracket. The maximum of −½x'Jx + x'b is ½ b'J^{−1}b,
with b = Λx_{t−1} + Df. Matching the coefficients of the quadratic guess gives:

    A_xx ← Λ − Λ J^{−1} Λ                       (Riccati equation; J contains A_xx)
    A_xf ← Λ J^{−1} D                           (linear in A_xf)
    A_ff ← D'J^{−1}D + β(I−Φ)'A_ff(I−Φ)
    const ← β (const + ½ tr(A_ff Cov(ε)))

* **Finite horizon** (`lq.finite_horizon`): start from V = 0 after the last period and apply
  these updates backwards. Each step also gives that period's policy (1).
* **Infinite horizon** (`lq.infinite_horizon`): the fixed point. It is computed independently of
  the recursion. A_xx solves a standard discrete algebraic Riccati equation (state x_{t−1},
  control Δx, stage cost ½[x'γΣx + 2x'γΣΔx + Δx'(γΣ+Λ)Δx], discount β), solved by
  `scipy.linalg.solve_discrete_are`. A_xf then solves a linear equation:
  vec(A_xf) = (I − β (I−Φ)' ⊗ ΛJ^{−1})^{−1} vec(ΛJ^{−1}B).

Why the DARE form is the same equation: with W = γΣ + βA, the DARE reads
A = W − W J^{−1} W. Since Λ = J − W, the right side of our Riccati equation is
Λ − ΛJ^{−1}Λ = (J−W) − (J−W)J^{−1}(J−W) = W − WJ^{−1}W. They are the same.

### 2.5 What the aim portfolio is

The Riccati equation can be rewritten as Λ J^{−1} = A_xx W^{−1}, because
A_xx = ΛJ^{−1}(J − Λ) = ΛJ^{−1}W. So A_xf = A_xx C, where aim = C f. Putting this into
aim_t = W^{−1}(B f_t + βA_xf(I−Φ) f_t):

    aim_t = Z · Markowitz_t + (I − Z) · E_t[aim_{t+1}],   Z = W^{−1} γΣ           (3)

Iterating (3) forward gives

    aim_t = Σ_{τ≥0} Z(I−Z)^τ E_t[Markowitz_{t+τ}]

The aim is a **weighted average of today's and expected future Markowitz portfolios**, with
geometrically declining weights. Because you will hold today's trade for a while, you aim at
where the target will be over your holding period, not just at today's target.

**One asset, K signals with decays ρ_k** (what the backtester uses, `lq.scalar_policy`). With
s = γσ² and scalar Λ = λ, the Riccati equation is a quadratic in A:

    β A² + (s + λ(1−β)) A − λ s = 0   →   A = [−(s+λ(1−β)) + √((s+λ(1−β))² + 4βλs)] / (2β)

Then the trading rate is M = A/λ, the aim weight is z = s/(s+βA), and each signal's Markowitz
portfolio is scaled by

    w_k = z / (1 − (1−z) ρ_k)

**Fast signals (small ρ_k) are scaled down more**, because they will be gone before the position
is built.

**Check against the paper's special case (memory, unverified).** As I recall it, GP discount
alpha and risk by (1−r)^{t+1} and costs by (1−r)^t. Dividing their objective by (1−r)^t maps it
onto ours with β = 1−r, γ → (1−r)γ, B → (1−r)B. With Λ = λΣ our solution then gives trading
rate a/λ, with a = [−(γ(1−r)+λr) + √((γ(1−r)+λr)² + 4γλ(1−r)²)] / (2(1−r)), and
aim = (γΣ)^{−1}B(I + aΦ/γ)^{−1} f. The test `test_consistent_with_recalled_gp_closed_form`
confirms this numerically. This is the closed form I remember from GP's Proposition; whether the
memory is exact must be checked against the paper.

### 2.6 Numerical validation (all in `tests/test_theory.py`, all pass)

| check | how | result |
|---|---|---|
| (i) Riccati = brute-force DP | one asset, horizon 4, grid 301 positions × 81 signal values, Gauss-Hermite expectation, exhaustive max | policy within one grid step (0.010 vs step 0.020); value within 0.04% of max. A deliberate ±10% error in Λ moves the policy by 0.07 and the value by 0.3, so the test has teeth |
| (ii) infinite = limit of finite | 3 assets, 2 signals, Λ not proportional to Σ; horizon 3000 recursion vs DARE solution | equal to 1e−9 |
| (ii') aim identity (3) | same problem | equal to 1e−10 |
| (ii'') closed form = matrix solver | 1 asset, 3 signals | equal |
| (iii) Monte Carlo | 2 assets, a fast (ρ=0.5) and a slow (ρ=0.98) signal, 200 paths × 2,000 periods, common random numbers | GP's utility is higher than naive and than the best of 5 EMA rates, by more than 3 standard errors |
| (iv) limits | Λ = 1e−9: M = I, aim = Markowitz. Λ = 1e9: M ≈ 0 | pass |

## 3. Proportional costs: the no-trade band

Real fees and spreads are **proportional**: a trade costs c|Δx|, not ½λΔx². There is no
closed form, so we solve one asset by dynamic programming on a grid (`dp.solve_band`).

**Normalisation.** If μ = σκg, where g is a unit-variance AR(1) signal with persistence ρ_g,
write x = (κ/(γσ)) z. The per-period reward x μ − (γ/2)σ²x² − c|Δx| becomes (κ²/γ) times

    z g − ½ z² − ĉ |Δz|,     ĉ = c / (σκ)

So one parameter, the normalised cost ĉ, describes the whole problem (plus ρ_g and β). ĉ is
"cost per unit traded divided by expected return per unit held per unit of signal". With fees
plus half-spread of 6.5 bp, hourly vol near 1% and an IC near 0.02, ĉ is about 3. That is
large: each trade costs about three hours of alpha.

**Band structure.** For each signal value g, let W(z) be the value of ending the period at
position z. W is concave in z: the reward is concave, and the concavity carries backwards through
the Bellman equation. Maximising W(z') − ĉ|z' − z| therefore has a three-region answer:

* **buy** up to L(g) = argmax W(z) − ĉz, if z < L(g);
* **sell** down to U(g) = argmax W(z) + ĉz, if z > U(g);
* **hold** (do nothing) if L(g) ≤ z ≤ U(g).

`dp.bellman_band` uses this to do each step in O(grid). `tests/test_band.py` checks it equals
the exhaustive O(grid²) search exactly. Value iteration runs until the span of the update is
below 1e−8. The span is used because the policy depends only on differences of V. Tests also
check that the band contains no-trade regions, widens with cost, and collapses to Markowitz
when ĉ = 0.

**How the band differs from the GP rule** (`results/fig_theory_band.png`, and
`results/theory_*.md`, simulated signal with half-life 24 periods):

1. **When to trade.** The band does nothing for small signal changes and then trades just
   enough to reach the band's edge. GP trades a fraction M of the gap every period. Under
   proportional costs, many small trades are expensive, so the band trades less often for the
   same tracking.
2. **Where to aim.** Both shrink towards zero relative to Markowitz, in the same direction (see
   panel b of the figure: the band's centre and the GP aim lie below Markowitz for g > 0). The
   band is not centred on Markowitz.
3. **Who wins.** Each policy is optimal for its own cost model. Under proportional costs the
   band beats GP (with GP's λ tuned) and EMA. Under quadratic costs GP beats the band (with its
   cost parameter tuned). The figures are in §4.

## 4. Simulation results (data-free)

Normalised single asset, signal half-life 24 periods, 100 paths × 4,000 periods, common random
numbers, β = 0.999. Utility is shown as % of the frictionless optimum (0.5 per period). Every
policy has its own parameter tuned by grid search: EMA rate, GP λ, band cost multiplier.
The standard errors are of the paired difference against naive. Full tables:
`results/theory_*.md`.

| true cost | naive | EMA (best rate) | GP (best λ) | band (best cost mult.) |
|---|---|---|---|---|
| proportional, ĉ = 0.5 | 79.0 | 85.3 | 85.7 | **87.4** (mult. 1.0) |
| proportional, ĉ = 2 | 21.8 | 64.9 | 66.3 | **71.5** (mult. 1.0) |
| quadratic, λ = 20 | −16.2 | 77.6 | **78.6** | 63.1 |

The standard error of each gain against naive is 0.02–0.23 percentage points, so every ranking
in the table is far outside noise. Three things to take away:

* Each optimal policy wins under its own cost model. The band's tuned cost multiplier came out
  at exactly 1, i.e. the DP's own answer is the best band. That is a self-consistency check.
* With **one** signal, GP is only slightly better than a well-tuned EMA, because both trade at
  a constant rate. GP's extra edge is its aim: it shrinks fast signals more than slow ones. That
  only matters with several signals of different speeds; validation test (iii) shows that case.
* **Decay sensitivity** (ĉ = 1, proportional cost). With a half-life of 3 periods, naive
  captures −3% of the frictionless utility (it loses money), GP 40% and the band 47%. With a
  half-life of 192 periods: naive 87%, GP 95%, band 96%. The faster the signal decays, the more
  optimal trading matters, and the less alpha is left to capture at all.

## 5. From theory to the data (approximations, stated plainly)

* **Diagonal risk and costs.** The backtest treats each asset separately: Σ diagonal (own
  volatility) and Λ diagonal (own impact). Then the n-asset problem splits into n one-asset
  problems with K signals each, solved in closed form (§2.5). This ignores correlations. That is
  defensible because the signals are cross-sectional (demeaned across assets), so the book is
  roughly market-neutral. But it is an approximation. The general matrix solver is built and
  tested, but not used on data.
* **Time-varying inputs.** Volatility, costs and the ridge coefficients change over time. The
  stationary solution assumes they are constant. We apply it each hour with the current values
  ("certainty-equivalent, re-solved each period"). That is standard practice, but it is not
  exactly optimal.
* **GP under proportional costs.** GP needs a quadratic cost. We give it
  Λ_policy = Λ_impact + m · 2c/q_ref, i.e. the impact term plus a quadratic proxy for the
  proportional cost that matches c|Δx| at trade size q_ref. The multiplier m is tuned on train.
  The backtest always charges the true cost c|Δx| + ½Λ_impactΔx².
* **The band uses one combined signal.** The DP needs a single AR(1) signal, so the band policy
  uses the ridge-combined score (persistence estimated on train). GP uses the six signals with
  their own decays. The band ignores the impact term when choosing trades, but pays it.
