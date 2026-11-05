# From signal to portfolio: one-page summary

**Question.** Can simple, honestly tested signals predict next-hour returns of the 25 most-traded
crypto perpetual futures? And if they can, what is the smartest way to trade them once trading
costs are counted?

**What we did.**
* We used public Binance data, hourly, 2021–2026. The universe is chosen month by month, delisted
  coins are kept, and non-crypto contracts are excluded.
* Six signals were fixed in advance: two momentum, short-term reversal, funding rate, volume shock
  and residual reversal. They were combined with a regression refitted every month on past data
  only.
* The resulting forecasts were traded four ways:
  1. **Naive:** rebuild the ideal portfolio every hour.
  2. **EMA:** move part of the way towards it each hour.
  3. **Garleanu-Pedersen:** the textbook optimal rule when trading costs grow with trade size
     squared. Trade a fixed fraction towards an "aim" portfolio that looks ahead to where the
     signals are going.
  4. **No-trade band:** the optimal rule for fees and spreads. Do nothing while your position is
     inside a band, and trade only to its edge.
* Settings were tuned on 2021–23, judged on 2024–mid-2025, and confirmed once on mid-2025 to
  October 2026.

**What we found.**
1. **The maths works.** The optimal rules were derived from scratch and checked four independent
   ways. In simulation, each wins exactly where theory says it should.
2. **The signals are real but tiny and fleeting.** Reversal over 1–4 hours predicts the next hour
   with high statistical confidence (t-statistics around 10). But the effect halves within one hour
   and is gone within six. A few extreme hours wipe it out in dollar terms, so even before costs
   the forecast made no money out of sample.
3. **Naive trading is a disaster.** Rebuilding the portfolio every hour costs about 1.6–2.2 times
   the capital each year in fees, spreads and market impact. The Sharpe ratio is −50.
4. **Smart trading avoids the disaster, mostly by not trading.** All three smart rules cut trading
   by about 99% and end near zero. Net Sharpe ratios fall between −1 and +1.5, with 95%
   confidence intervals that include zero, in both the test and the final holdout. After allowing
   for the number of variants tried, none is distinguishable from luck. Trials counted: 94 on
   test, 119 including the holdout, 139 including post-hoc checks. The no-trade band's
   Sharpe (+0.5 test, +0.9 holdout) should not be compared with the others. It comes from a
   near-empty book: on average 0.3% of capital, never 1% of capital in any test hour, and almost
   no trading.

**The lesson.** At retail taker costs (about 6.5 bp per trade plus impact), these hourly signals
are not worth trading. The value of the optimal-trading layer here is that it recognised this and
declined to trade, instead of turning a weak signal into a large loss. The conclusion depends on
the cost assumptions (spreads cannot be measured from this data). A post-hoc check (not
specified in advance; test period only) with costs of 1–6.5 bp per trade changes nothing. The 4-hour
reversal covered at most 0.75 bp per unit traded even in the period where it worked best, and it
lost money before costs out of sample.

**Why you can trust it (or check it).**
* Tests check for look-ahead in the signal and backtest code.
* PnL is reconciled with a hand calculation.
* Settings were fixed before the holdout; the exceptions are listed in report.md §8.
* The holdout was run exactly once.
* Every Sharpe ratio comes with a confidence interval.
* There is a selection-bias haircut for all trials.
* Two full reruns from raw data gave identical numbers and figures.
* One command rebuilds everything: `python -m s2p.run_all`.
