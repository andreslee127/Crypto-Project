# DATA

## Source

Public Binance Vision bulk files (https://data.binance.vision), USDT-margined futures,
monthly files. No API keys, no exchange account, no live connection.

| file | path pattern | used for |
|---|---|---|
| daily klines | `data/futures/um/monthly/klines/{SYM}/1d/{SYM}-1d-{YYYY-MM}.zip` | ranking the universe only |
| hourly klines | `.../klines/{SYM}/1h/{SYM}-1h-{YYYY-MM}.zip` | prices, volume, all signals, PnL |
| funding rates | `.../fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip` | funding signal and funding PnL |

Every file is verified against its published `.CHECKSUM` (SHA-256) and cached under
`data/raw/` with the same path. Failed requests are retried with exponential backoff. A 404 is
recorded in `data/raw/missing.json` rather than silently ignored.

```
python -m s2p.download        # ~25k small files on the first run; later runs use the cache
```

## Universe (point in time, survivorship-aware)

* The symbol list comes from the bucket listing. It includes delisted contracts, which Binance
  Vision keeps (e.g. LUNAUSDT, SRMUSDT, FTTUSDT). Delivery contracts (`_` in the name) and
  `...SETTLED` contracts are excluded. Both are filtered by name, so nothing about later
  performance enters.
* The universe for month m is the top 25 symbols by quote volume in month m−1, among symbols with
  at least 25 daily bars in m−1 and a bar on its last day. Ranking uses only data available at
  the start of month m (`data.rank_universe`, tested in `tests/test_data.py`).
* For each symbol, the hourly data of month m−1 is also loaded as a signal warm-up. The longest
  look-back is 504 hours (21 days), which is shorter than the shortest month (672 hours).
* **Delisting.** When a held symbol's data ends (gap longer than 3 hours), it is no longer
  tradeable. The position is closed at the last available price and the trading cost is
  charged. This is mildly optimistic: Binance announces delistings in advance, so a real trader
  would exit earlier and in an orderly way, but the final settlement price can differ from the
  last trade.

## Timestamps and alignment

* Bars are indexed by **close time** t = open_time + 1h, in UTC.
* Some recent Binance files use microseconds. Any value above 1e14 is treated as microseconds
  and converted (`data.to_ms`); this is reported in the quality summary.
* A decision at t uses closes ≤ t and earns the return from t to t+1.
* Funding settles on the hour (the exchange timestamp is floored to the hour). It is charged on
  the position held over the hour that ends at the settlement: payment = position × rate, so a
  long pays when the rate is positive.

## Quality checks (`data.load_panel`, fail loudly)

The run stops with `DataQualityError` and prints the failing items if any of these occur:

| check | rule |
|---|---|
| missing kline file | a needed (symbol, month) file does not exist |
| timestamps | open_time not on the hour; close − open ≠ 1h − 1ms; rows outside the file's month |
| duplicates | the same bar with different values (exact duplicates are dropped and reported) |
| OHLC / volume | high < max(open, close), low > min(open, close), price ≤ 0, volume < 0 |
| gaps | a gap > 24h inside a symbol's history; > 1% missing hours in an in-universe month |
| universe | a month without 25 qualifying symbols |
| funding | > 5% of in-universe symbol-months without a funding file |

These are reported as information only: short gaps, exact duplicates, microsecond files,
delisted symbols, |funding| > 1% per settlement, and hourly returns above 50%.
The summary from the last run is in `results/data_quality.md`.

## What kline data cannot tell you

There are no quotes or order-book data, so **spreads and market impact cannot be measured**. The
half-spread (2 bp) and the impact coefficient (Y = 1 in the square-root law) are assumptions.
Results are shown at 0×, 0.5×, 1×, 2× and 4× these levels. Volume and volatility, which kline
data does contain, only scale the impact across assets and over time.

## Holdout

Rows with decision time ≥ 2025-07-01 are removed inside `load_panel`, before anything else
touches them, unless `holdout=True`. Only `python -m s2p.run_all --holdout` sets that flag. It
writes `results/HOLDOUT_RUN.txt` and refuses to run a second time.
