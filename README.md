# From signal to portfolio

**Finding short-horizon alpha in crypto perpetuals, and trading it optimally under transaction costs.**

The pipeline has two layers:

1. **Alpha research.** Six pre-registered signals on the 25 most-traded Binance USDT perpetuals
   (hourly bars, 2021–2026) are tested honestly: point-in-time universe, walk-forward fitting,
   multiple-testing control, and a holdout touched once.
2. **Optimal implementation** (the centrepiece). The resulting forecasts are traded four ways.
   The policies are naive rebalancing, an EMA-smoothed target, the Garleanu-Pedersen (2013)
   dynamic policy, and a proportional-cost no-trade band solved by dynamic programming. They are
   compared on net Sharpe, turnover and cost drag, with block-bootstrap confidence intervals.

Results: [`report.md`](report.md) (full), [`SUMMARY.md`](SUMMARY.md) (one page).
Derivations: [`THEORY.md`](THEORY.md). Data: [`DATA.md`](DATA.md).

## Reproduce

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, matplotlib, pyyaml, pytest (pinned)
python -m pytest -q                 # data-free and synthetic tests (also run in CI)
python -m s2p.download              # public Binance Vision files, cached and checksum-verified
python -m s2p.run_all               # every table and figure in results/ (about 10-20 minutes)
python -m s2p.run_all --holdout     # the final, one-off holdout evaluation
python -m s2p.run_all --synthetic   # optional: whole pipeline on synthetic data (results/synthetic/)
```

Everything is driven by [`config.yaml`](config.yaml): signals, dates, cost assumptions, tuning
grids and random seed. It was fixed before any real-data result was seen.

## Layout

| file | what it does |
|---|---|
| `s2p/lq.py` | Riccati / Garleanu-Pedersen solution (finite, infinite horizon, closed forms) |
| `s2p/dp.py` | grid dynamic programmes: brute-force check, proportional-cost no-trade band |
| `s2p/sim.py`, `s2p/theory_report.py` | Monte Carlo on simulated signals, theory tables and figure |
| `s2p/download.py`, `s2p/data.py` | downloader; parsing, universe, holdout cut, quality checks |
| `s2p/signals.py`, `s2p/research.py` | signals; IC, decay, Holm; walk-forward ridge |
| `s2p/costs.py`, `s2p/backtest.py` | cost model; the four policies and PnL accounting |
| `s2p/stats.py` | Sharpe CIs (block bootstrap), Newey-West, deflated Sharpe, trial log |
| `s2p/run_all.py` | the one command |
| `tests/` | theory validation, band DP, look-ahead, PnL reconciliation, data checks, stats |

## Version history

The project was not under version control while it was built. **The first git commit was made
after the holdout had already been run**, so git history cannot show that the pre-registration
in `config.yaml` came before the results. The comments in `config.yaml` and `report.md` §8 list
every choice made after seeing data.

## Ground rules

No live trading and no API keys. No look-ahead (enforced by `tests/test_lookahead.py`).
Parameters are tuned on train only. The holdout is cut at load time and run once. Every
variant tried is counted. Anything made from synthetic data is labelled SYNTHETIC.
