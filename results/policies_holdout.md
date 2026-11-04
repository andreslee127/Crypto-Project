## Selected policies, holdout window, 1x costs

Returns are PnL / capital (USD 10m, no compounding). SR CI: circular block bootstrap, 1-week blocks.

| policy | param | net_SR | SR_lo95 | SR_hi95 | gross_SR | net_ret_pct | gross_ret_pct | vol_pct | cost_drag_pct | impact_pct | funding_pct | turnover_x_per_yr | gross_lev | n_bars | n_eff | n_blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive |  | -48.967 | -52.007 | -46.106 | 0.102 | -161.121 | 0.320 | 3.290 | 162.802 | 108.226 | -1.362 | 839.632 | 0.151 | 11616 | 11616.000 | 69 |
| EMA | 0.005 | -0.977 | -3.000 | 1.354 | -0.650 | -1.752 | -1.160 | 1.793 | 0.460 | 0.042 | 0.131 | 6.433 | 0.054 | 11616 | 11351.051 | 69 |
| Garleanu-Pedersen | 16.000 | 0.614 | -0.957 | 2.058 | 0.186 | 0.841 | 0.254 | 1.369 | 0.438 | 0.048 | -1.025 | 5.991 | 0.049 | 11616 | 11581.118 | 69 |
| No-trade band | 2.000 | 0.924 | -0.415 | 2.346 | 0.099 | 0.188 | 0.020 | 0.203 | 0.004 | 0.001 | -0.171 | 0.046 | 0.003 | 11616 | 11584.968 | 69 |
