## Selected policies, test window, 1x costs

Returns are PnL / capital (USD 10m, no compounding). SR CI: circular block bootstrap, 1-week blocks. Read the no-trade band's Sharpe with its gross_lev (average gross exposure / capital), turnover and time in market (time_in_market = share of hours with any position; time_above_1pct = share with gross exposure >= 1% of capital): a Sharpe on a near-empty book is not comparable to the other policies.

| policy | param | net_SR | SR_lo95 | SR_hi95 | gross_SR | net_ret_pct | gross_ret_pct | vol_pct | cost_drag_pct | impact_pct | funding_pct | turnover_x_per_yr | gross_lev | time_in_market | time_above_1pct | n_bars | n_eff | n_blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive |  | -53.572 | -58.111 | -49.131 | -0.907 | -225.722 | -3.551 | 4.213 | 223.022 | 155.577 | -0.851 | 1037.616 | 0.190 | 1.000 | 1.000 | 13128 | 13012.151 | 78 |
| EMA | 0.005 | 1.479 | -0.285 | 3.234 | 1.539 | 2.485 | 2.586 | 1.680 | 0.589 | 0.062 | -0.488 | 8.112 | 0.069 | 1.000 | 1.000 | 13128 | 13033.432 | 78 |
| Garleanu-Pedersen | 16.000 | -0.162 | -1.804 | 1.464 | -0.080 | -0.252 | -0.123 | 1.551 | 0.664 | 0.064 | -0.535 | 9.231 | 0.062 | 1.000 | 1.000 | 13128 | 12659.688 | 78 |
| No-trade band | 2.000 | 0.506 | -0.750 | 1.786 | 0.538 | 0.085 | 0.091 | 0.169 | 0.003 | 0.001 | 0.003 | 0.035 | 0.003 | 1.000 | 0.000 | 13128 | 13128.000 | 78 |
