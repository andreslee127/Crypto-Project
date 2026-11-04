# From signal to portfolio: short-horizon alpha in crypto perpetuals and how to trade it

*Research report. Every number below comes from `results/` and is rebuilt by
`python -m s2p.run_all` (holdout: `--holdout`, run once). The derivations are in
[THEORY.md](THEORY.md) and the data handling in [DATA.md](DATA.md).*

## Summary

* **The theory works exactly as derived.** The Garleanu-Pedersen (GP) solution matches
  brute-force dynamic programming, its infinite-horizon limit and its limiting cases. In
  simulation it beats naive and EMA trading under quadratic costs. A proportional-cost no-trade
  band, solved by dynamic programming, beats GP when costs are proportional, as theory says it should.
* **The signals predict next-hour returns in a statistical sense, but not in a tradeable one.**
  Three of six pre-registered signals have test-period ICs that survive Holm correction: 4-hour
  reversal, residual reversal and funding. The predictive power of the strongest two halves within
  one hour and is gone within six. Their rank ICs (0.02–0.03) overstate the linear relation
  (Pearson IC on vol-scaled returns 0.008–0.009). In dollars, a few extreme hours dominate: even
  before costs, the combined forecast loses money in the test period (gross Sharpe −0.89, 95% CI
  [−2.59, 0.68]).
* **Under naive trading, no alpha survives costs.** Rebalancing fully every hour costs about 220%
  of capital a year (test), giving a net Sharpe of −54. The train-period gross alpha would only
  break even at a cost of about 0.9 bp per unit traded. The assumed cost is 6.5 bp plus impact.
* **Optimal trading removes almost all the damage, but finds no net alpha.** EMA, GP and band
  policies tuned on train all cut turnover by about 99% and end near zero. Test net Sharpes are EMA
  +1.48 [−0.29, 3.23], GP −0.16 [−1.80, 1.46] and band +0.51 [−0.75, 1.79]. None is
  distinguishable from zero, and none survives the selection-bias haircut for 74 trials. The
  train-optimal choice for each policy was to trade as little as its grid allowed.
* **The holdout (run once, July 2025 – October 2026) agrees.** Naive −49; EMA −0.98, GP +0.61 and
  band +0.92, all with CIs containing zero. The GP-vs-EMA ranking flipped relative to test, and the
  funding signal faded (§7).

The honest headline: **with retail-level taker costs, these hourly signals are not worth trading,
and the job of the optimal-trading layer is mainly to recognise that.**

---

## 1. Question and design

Research questions (from the brief):

1. Do the signals predict returns out of sample (IC, ICIR, decay, turnover-adjusted Sharpe, after
   multiple-testing control)?
2. How much gross alpha survives realistic costs under naive trading?
3. How much do EMA, Garleanu-Pedersen and no-trade-band policies recover?
4. How sensitive are the conclusions to cost levels and signal decay?

**Data.** Binance USDT-margined perpetuals, hourly bars and funding rates from the public Binance
Vision files. All files are checksum-verified. The universe for each month is the 25 most-traded
contracts of the previous month, chosen point in time. Delisted contracts (LUNA, MATIC, EOS and
others) stay in, so there is no survivorship bias. 191 distinct symbols appear before the holdout
and 264 overall. Non-crypto contracts (tokenised stocks and commodities, all from 2026) are
excluded by a rule based on the exchange's own classification.

**Splits** (fixed in `config.yaml` before any result):

| period | dates | role | hourly decisions |
|---|---|---|---|
| burn-in | 2021-01 – 2021-06 | fit only | – |
| train | 2021-07 – 2023-12 | tuning; all parameters chosen here | 21,936 |
| test | 2024-01 – 2025-06 | evaluation | 13,128 (78 weekly blocks) |
| holdout | 2025-07 – 2026-10-27 | one final run | see §7 |

**Signals** (pre-registered; each is a cross-sectional rank, using data ≤ t, paired with the
return from t to t+1):

| name | definition | hypothesis |
|---|---|---|
| mom_7d / mom_21d | past 7- / 21-day log return | momentum (+) |
| rev_4h | minus past 4-hour return | short-term reversal (+) |
| funding | minus funding paid by longs over 24 h | crowded longs underperform (+) |
| volume_shock | log(24 h volume / 21-day average) | no prior (two-sided) |
| resid_rev | minus vol-scaled 24 h return net of market beta | idiosyncratic reversal (+) |

**From signals to expected returns.** A ridge regression of next-hour vol-scaled return
(demeaned across assets, winsorised at ±5) on the six signals. It is refitted every month on all
earlier data. The penalty (α = 0.1) was chosen by out-of-sample IC on train. The expected return
is μ = σ · score. Each signal's decay is its pooled AR(1) coefficient on train.

**Capital and returns.** Capital is a fixed USD 10m; returns are PnL / capital with no
compounding. Risk aversion is set so the naive (Markowitz) target has an ex-ante volatility of 10%
a year on train. Net PnL = gross PnL − linear costs − impact − funding, checked against a
hand-computed example (`tests/test_backtest.py`).

**Costs (assumptions; see §8).** Taker fee 4.5 bp plus half-spread 2 bp per unit traded, plus
quadratic impact ½λΔx². λ is calibrated so that, at a USD 25k trade, it equals the square-root
law Y·σ_daily·√(Q/ADV) with Y = 1. Kline data contain no quotes, so these levels cannot be
measured and are varied from 0× to 4×.

## 2. The theory layer (data-free)

Full derivation and tables: THEORY.md. In brief:

* The optimal rule is **x_t = x_{t−1} + M(aim_t − x_{t−1})**. M comes from a Riccati equation,
  and the aim is a geometrically weighted average of today's and expected future Markowitz
  portfolios. For one asset, fast-decaying signals are scaled down by z / (1 − (1−z)ρ_k).
* **Validation (all tests pass):**
  * The finite-horizon Riccati policy is within one grid step of brute-force DP. A 10% error in Λ
    is detected.
  * The infinite-horizon DARE solution equals a 3,000-step backward recursion to 1e−9.
  * In Monte Carlo, GP beats naive and the best EMA by more than 3 standard errors.
  * Λ → 0 gives Markowitz; Λ → ∞ gives no trading.
  * The solution also matches the GP closed form for Λ = λΣ as I recall it. The paper itself was
    not consulted (fetching it was declined), so that check rests on memory.
* **No-trade band:** a DP on a (position, signal) grid. Its structure is buy / hold / sell
  (`results/fig_theory_band.png`). In simulation with a proportional cost ĉ = 2 (normalised;
  realistic for this data), the band captures 71.5% of the frictionless utility, GP 66.3%, EMA
  64.9% and naive 21.8%. Under quadratic costs the ranking reverses between band and GP (GP 78.6%,
  band 63.1%). Both shrink the position towards zero relative to Markowitz. The difference is that
  GP trades a little every period, while the band trades only when it leaves the band, and then
  only to its edge.

## 3. RQ1 — Do the signals predict returns out of sample?

Test window (train value in brackets). The IC is the cross-sectional Spearman correlation with
the next-hour return. t uses Newey-West with 48 lags, and p is Holm-adjusted over six signals.
The gross Sharpe is for a stand-alone, vol-weighted, gross-leverage-1 portfolio, before costs,
with a block-bootstrap 95% CI.

| signal | IC test [train] | Pearson IC (vol-scaled) | t (NW) | Holm p | IC at h=2 / 24 | signal half-life | gross SR test [95% CI] | turnover ×/yr | break-even bp |
|---|---|---|---|---|---|---|---|---|---|
| mom_7d | −0.006 [−0.014] | 0.001 | −2.4 | 0.050 | −0.004 / +0.003 | 41 h | −0.28 [−2.03, 1.48] | 1,141 | −0.5 |
| mom_21d | +0.002 [−0.010] | 0.007 | 0.9 | 0.76 | +0.004 / +0.008 | 92 h | 0.28 [−1.16, 1.83] | 708 | 0.9 |
| rev_4h | **+0.028** [0.054] | 0.008 | 11.8 | <1e−30 | +0.018 / −0.010 | 1.7 h | −0.85 [−2.33, 0.75] | 5,999 | −0.3 |
| funding | **+0.009** [0.005] | 0.003 | 4.2 | 1e−4 | +0.009 / +0.013 | 41 h | −0.48 [−2.01, 1.00] | 513 | −1.7 |
| volume_shock | −0.002 [−0.008] | −0.001 | −0.8 | 0.76 | −0.001 / +0.008 | 86 h | 0.10 [−1.52, 1.80] | 719 | 0.3 |
| resid_rev | **+0.019** [0.033] | 0.009 | 9.0 | 1e−18 | +0.014 / −0.004 | 9.2 h | −0.47 [−2.08, 1.27] | 2,624 | −0.4 |
| ridge combination | +0.024 [0.044] | 0.007 | 10.5 | – | +0.016 / −0.001 | 2.5 h | −0.89 [−2.59, 0.68] | 5,581 | −0.3 |

ICIR (annualised mean/std of hourly IC), test: rev_4h 9.2, resid_rev 7.1, funding 3.5, the rest
below 2. Full tables: `results/signals_test.md`, `signals_train.md`, `combined_test.md`.

**Answer.** *Statistically yes, economically no.*

* **Reversal at 1–4 hours** is the only strong effect. It is significant in both periods but
  roughly halves from train to test. Most of it is gone after one bar: the IC falls from 0.028 to
  0.018 if you skip one hour, and to 0.002 at six hours. This pattern is what bid-ask bounce in
  last-trade closes, plus very fast liquidity-driven reversal, would produce. Kline data cannot
  separate the two. Either way, it is not capturable at hourly frequency once you pay a spread.
* **Funding** is small but persistent, and it is the one signal whose IC does not decay within 24
  hours (0.009 → 0.013).
* **Momentum** went the wrong way: 7- and 21-day "momentum" acted as reversal on train, and
  neither has a stable sign. Volume shock is noise out of sample.
* **The rank IC overstates.** The Pearson IC on vol-scaled returns, the relation the ridge
  actually fits, is about a third of the Spearman IC. And **tails decide the dollars**: if hourly
  returns are clipped at ±5% (a diagnostic, not a tradeable number), rev_4h's test gross Sharpe
  goes from −0.85 to +0.89, and the combination's from −0.89 to +0.61. The forecasts are right
  slightly more often than not, but wrong in the few hours that matter most. Shorting coins that
  have just pumped is costly when they keep pumping.
* **Decay half-lives** (AR(1) of the signal, the input to GP): 1.7 h (rev_4h), 9 h (resid_rev),
  41 h (mom_7d, funding), 86–98 h (volume, mom_21d). The combined forecast has a half-life of only
  2.5 h because the ridge loads mostly on the fast reversal signals.

## 4. RQ2 — How much survives naive trading?

Naive trading rebalances fully to the Markowitz target every hour.

| | train | test |
|---|---|---|
| gross Sharpe (0× costs) | 2.65 | −0.69 [−2.29, 0.94] |
| net Sharpe (1× costs) | −85.0 [−90.7, −79.7] | −53.6 [−58.1, −49.1] |
| gross return, % of capital / yr | 20.3 | −3.6 |
| cost drag, % / yr (of which impact) | 850 (707) | 223 (156) |
| turnover, × capital / yr | 2,205 | 1,038 |

**Answer: none.** The linear costs alone (fee + spread, about 67–143% a year) are several times the
gross alpha in either period. On train, the combined forecast's break-even cost is 0.94 bp per
unit traded, against an assumed 6.5 bp. The quadratic impact makes it worse still. For very large
trades the quadratic overstates the square-root law, but the conclusion does not depend on it.
In the test period there is no gross alpha to lose.

## 5. RQ3 — How much does optimal trading recover?

Test window, 1× costs. Every parameter was chosen by net Sharpe on train. n = 13,128 hourly
observations; effective n (autocorrelation-adjusted) = 12,700–13,100; 78 weekly bootstrap blocks.

| policy | train-chosen parameter | net SR [95% CI] | gross SR | net % / yr | vol % / yr | turnover ×/yr | cost drag % / yr | funding % / yr |
|---|---|---|---|---|---|---|---|---|
| naive | – | −53.6 [−58.1, −49.1] | −0.91 | −225.7 | 4.2 | 1,038 | 223.0 | −0.85 |
| EMA | rate 0.005 | **+1.48** [−0.29, 3.23] | 1.54 | +2.5 | 2.6 | 8.1 | 0.59 | −0.49 |
| Garleanu-Pedersen | cost mult. 16 | −0.16 [−1.80, 1.46] | −0.08 | −0.3 | 1.6 | 9.2 | 0.66 | −0.54 |
| no-trade band | cost mult. 2 | +0.51 [−0.75, 1.79] | 0.54 | +0.1 | 0.2 | 0.04 | 0.00 | 0.00 |

(Negative funding = funding received.)

Paired block-bootstrap differences in net Sharpe (same resampled weeks for both policies):

| comparison | Δ net SR | 95% CI | share of draws > 0 |
|---|---|---|---|
| EMA − naive | +55.1 | [50.6, 59.4] | 100% |
| GP − naive | +53.4 | [49.7, 57.6] | 100% |
| band − naive | +54.1 | [49.6, 58.6] | 100% |
| GP − EMA | −1.64 | [−2.72, −0.54] | 0.4% |
| band − GP | +0.67 | [−1.00, 2.35] | 79% |

**Selection-bias haircut.** 74 variants were evaluated in total (`results/trials_test.csv`: 6
signals, 4 ridge penalties, 20 policy-grid points, 5 grid-edge checks, 32 cost scenarios, 3 decay
scenarios, 4 main evaluations). The expected maximum Sharpe of 74 pure-noise strategies over this
sample is about 2.0. The deflated-Sharpe probability that the true Sharpe exceeds that is 0.27
(EMA), 0.04 (band) and 0.004 (GP).

**Answer.**

* *Against naive*, optimal trading recovers essentially everything: about 225% of capital a year
  in avoided costs. The difference is overwhelming and certain.
* *In absolute terms* it recovers nothing that is statistically real. All three policies end with
  net Sharpes whose CIs contain zero, and none passes the haircut.
* **They do this by not trading.** On train, the best EMA rate was the smallest on the grid. The
  best GP cost multiplier was the largest, and the band's best choice kept it nearly flat (0.3%
  gross exposure in test). The post-hoc grid-edge check (`results/tuning_edge_check.md`)
  confirms the train optimum keeps moving towards "trade less": GP's train Sharpe rises from 0.21
  to 0.32 as the multiplier goes to 128. That is what a correctly working optimiser does when the
  alpha is smaller than the cost of capturing it.
* **GP vs EMA.** On test, GP is worse than EMA (−1.64, CI excludes zero). On train the order is
  reversed (GP 0.21, EMA −0.43). I do not read this as evidence against GP. A likely explanation,
  not tested here: GP scales the fast reversal signals down and leans on the slow ones (funding,
  momentum, volume), whose train coefficients did not hold up in test. EMA smoothing of a
  reversal-dominated target behaves like a slower, weekly reversal, which happened to work in
  2024–25. A ranking that flips between periods, among near-zero Sharpes, after 74 trials, is noise
  until shown otherwise.

## 6. RQ4 — Sensitivity to costs and decay

Net Sharpe on test. Policies keep their train-chosen parameters, but see and pay the scaled
costs, so GP and the band adapt automatically.

| cost multiplier | naive | EMA | GP | band |
|---|---|---|---|---|
| 0× | −0.69 | 1.83 [0.10, 3.53] | −0.69 | −0.69 |
| 0.5× | −28.6 | 1.65 | −0.11 | −0.30 |
| 1× | −53.6 | 1.48 | −0.16 | 0.51 |
| 2× | −90.0 | 1.13 | −0.32 | 0.28 |
| 4× | −123.0 | 0.43 | −0.58 | no trades |
| fee ×2 / spread ×2 / impact ×4 (others 1×) | −63.5 / −58.1 / −104.9 | 1.26 / 1.38 / 1.37 | −0.26 / −0.20 / −0.22 | 0.46 / 0.87 / 0.50 |

`results/fig_cost_sensitivity_test.png` plots these with CIs.

* **Costs.** Naive is destroyed at any cost above zero, and its loss scales linearly with the cost
  level. The optimal policies are almost insensitive because they respond by trading less. At 0×
  costs, GP and band collapse to naive (with nothing to save, they trade the same way). This shows
  the test-period forecast itself had no gross value (−0.69). EMA's 0× result (1.83, CI just above
  zero) is the one interesting number: the smoothed (slower) forecast had gross value in test. It
  had none on train (gross −0.58), so it is not a robust finding.
* **Decay.** Giving GP the wrong half-lives (×0.5 or ×2) changes its test Sharpe by less than 0.1
  (−0.21 / −0.16 / −0.14). That is invisible next to a CI of ±1.6. In simulation
  (THEORY.md §4), decay matters a lot. With a 3-period half-life, naive captures −3% of the
  frictionless utility and the band 47%. With a 192-period half-life, naive captures 87% and the
  band 96%. Fast signals make optimal trading essential and leave little to capture. That is the
  situation the data put us in.

## 7. Holdout (run once)

The holdout runs from 2025-07-01 to 2026-10-27: 11,616 hourly decisions in 69 weekly blocks. It
was loaded only with `--holdout` and run once, with every parameter frozen from train. The run
wrote `results/HOLDOUT_RUN.txt`, and a second run is refused. A first attempt stopped at the
data-quality check before computing anything, because 2026-10-28 is missing at the source. The
end date was then moved to 2026-10-27 and the run repeated. Nothing else changed. Results are
reported as they came out.

| policy | net SR [95% CI] | gross SR | net % / yr | turnover ×/yr | cost drag % / yr |
|---|---|---|---|---|---|
| naive | −49.0 [−52.0, −46.1] | 0.10 | −161.1 | 840 | 162.8 |
| EMA | −0.98 [−3.00, 1.35] | −0.65 | −1.8 | 6.4 | 0.46 |
| Garleanu-Pedersen | +0.61 [−0.96, 2.06] | 0.19 | +0.8 | 6.0 | 0.44 |
| no-trade band | +0.92 [−0.42, 2.35] | 0.10 | +0.2 | 0.05 | 0.00 |

| comparison | Δ net SR [95% CI] |
|---|---|
| each smart policy − naive | +48 to +50, all CIs far above 0 |
| GP − EMA | +1.59 [−0.26, 3.27] (it was −1.64 on test) |
| band − GP | +0.31 [−1.27, 1.85] |

**Signals in the holdout.**
* rev_4h (IC 0.029, t = 10.7) and resid_rev (0.014, t = 5.5) are still significant.
* **funding has faded** (0.002, n.s.).
* volume_shock is significant by rank IC (0.011, t = 4.8), but its Pearson IC is negative
  (−0.002), another case where rank and dollar relations disagree. mom_21d is borderline (Holm
  p = 0.03).
* The combined forecast has IC 0.028 (Pearson 0.008) and a gross Sharpe of −0.10 [−1.62, 1.43].
  Again, there is no gross value in dollars.

**Haircut.** Counting the 45 holdout evaluations, 119 variants have been evaluated. The noise
maximum is a Sharpe of about 2.25, and no policy's deflated-Sharpe probability exceeds 0.07.

**Cost and decay sensitivity (holdout).** At 0× costs naive has a Sharpe of +0.53
[−0.91, 2.04]: no significant gross alpha. GP stays between 0.0 and 0.8 at every cost level
and under both wrong half-life assumptions.

**Reading.** The holdout confirms the test conclusions. Naive trading is ruinous. The optimal
policies avoid that by trading very little, and their net Sharpes cannot be told apart from zero
or from each other. The GP-vs-EMA ranking flipped sign between test and holdout, which supports
reading it as noise. Most of EMA's holdout loss is a single drop of about 2.5% of capital in early
September 2025 (`results/fig_pnl_holdout.png`). It is the same tail-dominance seen in §3: one bad
hour decides the period. One honest caveat: the band's +0.92 comes from holding a tiny book (0.3% gross
exposure). A Sharpe on that is meaningless as an investment result.

## 8. Limitations

* **Costs are assumptions.** Kline data have no bid/ask, so the 2 bp half-spread and Y = 1 impact
  are guesses. The fee (4.5 bp) is the retail taker rate. A market maker, or a VIP account with
  maker rebates, pays far less, and §6 shows that only near-zero costs could make naive trading of
  the fast signals interesting. The quadratic impact matches the square-root law only at the
  USD 25k reference size: it overstates large trades (naive) and understates tiny ones. Roll's
  spread estimator was tried and is not usable at hourly frequency: own-asset autocorrelation is
  negative for only 35–46% of symbols.
* **Execution at the close.** Decisions and fills both use the last-trade close of the hour. That
  is optimistic for reversal signals (bid-ask bounce) and ignores latency. The half-spread charge
  covers part of this.
* **Model simplifications** (THEORY.md §5). Risk and impact matrices are diagonal; the book is not
  forced to be market-neutral. The stationary GP solution is applied with time-varying inputs. GP
  sees proportional costs only through a tuned quadratic proxy, and the band uses a single
  combined signal and ignores impact when deciding.
* **Crypto-specific risks not modelled:** exchange failure (FTX 2022), auto-deleveraging,
  funding-rate caps, delisting announcements (we exit at the last traded price), and tail events
  in newly listed coins.
* **Statistical power.** Eighteen months of test gives Sharpe CIs about ±1.7 wide. A true net
  Sharpe below about 1.5 could not be detected, and differences between good policies are mostly
  below that.
* **Decisions taken after seeing data** (all disclosed in `config.yaml`):
  * Before any result: a finer band-DP grid; a wider normalised-cost grid; excluding the 15
    non-crypto contracts after seeing the universe list.
  * After the first real run: an IC at h = 2 as a bounce diagnostic; Pearson-IC and
    clipped-return diagnostic columns; the grid-edge check (train-scored, counted as trials;
    selections unchanged).
  * Before the holdout result: ending the holdout on 2026-10-27, because Binance Vision has no data
    for 2026-10-28.
  * None of these changed a parameter used in the test evaluation.
* **Engineering incidents, disclosed:**
  * Some Binance monthly files are truncated (e.g. 12 symbols end on 26 Feb 2022). They were
    repaired from the daily files (80 days), and the quality check enforces agreement.
  * Halted contracts with zero volume produced infinite impact. They are now untradeable, and
    exits are charged the last valid λ.
  * During development, identical computations intermittently returned NaN in some processes. The
    cause was not isolated; pandas' numexpr and bottleneck paths are the prime suspects. They are
    now disabled, guards fail loudly on non-finite estimates, and **two complete pipeline runs
    produced identical results in all 18 tables**.

## 9. What could invalidate this

1. **Cheaper execution.** If realistic costs for this strategy are 1–2 bp (maker fills, rebates,
   or better spread estimates), the fast reversal could become tradeable, and the "do not trade"
   conclusion would be about retail costs, not about the signals.
2. **Price measurement.** If last-trade closes systematically differ from tradeable prices (mid
   or bid/ask), every IC here, especially for reversal, may be biased. Order-book data would
   settle it.
3. **A bug we did not test for.** The tests cover look-ahead, accounting, the solvers and data
   checks, not everything. A result this negative is less suspicious than a positive one. An
   independent re-implementation of the backtester is still the strongest check available.
4. **Regime dependence.** 2021 (strong reversal) dominates train. The relation weakened in
   2024–25, and the momentum signs flipped. A different sample could change which slow signals the
   ridge favours, and with it the GP/EMA comparison.
5. **The haircut.** If the true number of effective trials is much lower than 74 (many are highly
   correlated variants), the haircut is too harsh. Even without it, no policy's CI excludes zero.

**Figures** (`results/`):
* `fig_theory_band.png`: no-trade band vs the GP aim.
* `fig_ic_test.png`, `fig_ic_holdout.png`: signal ICs with CIs.
* `fig_frontier_train.png`: net Sharpe vs turnover across the tuning grids.
* `fig_pnl_test.png`, `fig_pnl_holdout.png`: cumulative net PnL.
* `fig_cost_sensitivity_*.png`: cost sensitivity.

## 10. Conclusion

The optimal-trading machinery is correct and does what theory says. In simulation, each policy
wins exactly where it should. On real hourly crypto data, the pre-registered signals carry real
but very short-lived information, smaller than the cost of trading it. Naive implementation turns
that information into large losses. Optimal implementation turns it into roughly nothing, by
declining to trade. That is a negative result about the signals, and a positive one about the value
of treating implementation as part of the research rather than an afterthought.
