## SYNTHETIC DATA - pipeline check only. Selected policies, test window, 1x costs

Returns are PnL / capital (USD 10m, no compounding). SR CI: circular block bootstrap, 1-week blocks.

| policy | param | net_SR | SR_lo95 | SR_hi95 | gross_SR | net_ret_pct | gross_ret_pct | vol_pct | cost_drag_pct | impact_pct | funding_pct | turnover_x_per_yr | gross_lev | n_bars | n_eff | n_blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive |  | 2.998 | 1.378 | 4.635 | 8.890 | 25.021 | 73.878 | 8.347 | 51.441 | 17.879 | -2.584 | 516.332 | 0.367 | 13128 | 13128.000 | 78 |
| EMA | 0.100 | 6.828 | 5.303 | 8.300 | 7.909 | 49.964 | 57.860 | 7.317 | 10.118 | 0.799 | -2.221 | 143.362 | 0.322 | 13128 | 12897.321 | 78 |
| Garleanu-Pedersen | 2.000 | 7.097 | 5.584 | 8.634 | 7.877 | 37.387 | 41.489 | 5.268 | 5.600 | 0.252 | -1.497 | 82.271 | 0.225 | 13128 | 12997.101 | 78 |
| No-trade band | 2.000 | 6.741 | 5.226 | 8.224 | 7.192 | 22.861 | 24.384 | 3.391 | 2.336 | 0.545 | -0.813 | 27.561 | 0.139 | 13128 | 13030.649 | 78 |
