# Notes for AI assistants working on this repo

Research project: short-horizon alpha on Binance USDT perpetuals (layer 1) traded with
Garleanu-Pedersen dynamic trading vs naive / EMA / no-trade-band policies (layer 2).

Rules that must not be broken:
- No look-ahead. A decision at bar close t uses data with timestamp <= t only.
  `tests/test_lookahead.py` enforces this; keep it passing.
- `config.yaml` is pre-registered. Do not change signals, splits or grids silently; any change
  after results were seen goes in report.md and counts as an extra trial.
- The holdout (see `splits.holdout`) is dropped inside `s2p.data.load_panel` unless
  `holdout=True`. Only `python -m s2p.run_all --holdout` sets it, and it may run once.
- Parameters are tuned on train only. Every variant tried is recorded via `s2p.stats.TrialLog`.
- Dependencies: numpy, pandas, scipy, matplotlib (+ pyyaml, pytest). No other libraries.
- Anything built from synthetic data must be labelled SYNTHETIC.
- No live trading, no API keys.

Layout: `s2p/lq.py` (Riccati / GP theory), `s2p/dp.py` (grid dynamic programmes),
`s2p/download.py`, `s2p/data.py`, `s2p/signals.py`, `s2p/research.py`, `s2p/costs.py`,
`s2p/backtest.py`, `s2p/stats.py`, `s2p/run_all.py`. Tests in `tests/`.

Commands: `python -m pytest -q`; `python -m s2p.download`; `python -m s2p.run_all`.
