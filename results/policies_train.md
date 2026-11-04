## Selected policies on train (in-sample for tuning)

Read the no-trade band's Sharpe with its gross_lev (average gross exposure / capital), turnover and time in market (time_in_market = share of hours with any position; time_above_1pct = share with gross exposure >= 1% of capital): a Sharpe on a near-empty book is not comparable to the other policies.

| policy | net_SR | SR_lo95 | SR_hi95 | gross_SR | net_ret_pct | gross_ret_pct | vol_pct | cost_drag_pct | impact_pct | funding_pct | turnover_x_per_yr | gross_lev | time_in_market | time_above_1pct | n_bars | n_eff | n_blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | -84.987 | -90.686 | -79.691 | 2.651 | -827.334 | 20.285 | 9.735 | 849.960 | 706.659 | -2.341 | 2204.632 | 0.397 | 1.000 | 1.000 | 21936 | 510.319 | 130 |
| EMA | -0.427 | -1.593 | 0.721 | -0.576 | -1.119 | -1.503 | 2.620 | 1.352 | 0.248 | -1.735 | 16.977 | 0.127 | 1.000 | 1.000 | 21936 | 21540.910 | 130 |
| Garleanu-Pedersen | 0.213 | -0.951 | 1.381 | 0.100 | 0.585 | 0.274 | 2.745 | 1.481 | 0.266 | -1.791 | 18.682 | 0.119 | 1.000 | 1.000 | 21936 | 21755.038 | 130 |
| No-trade band | 0.016 | -0.821 | 1.017 | 0.090 | 0.021 | 0.115 | 1.266 | 0.353 | 0.189 | -0.259 | 2.516 | 0.017 | 1.000 | 0.382 | 21936 | 21936.000 | 130 |
