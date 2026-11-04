## SYNTHETIC DATA - pipeline check only. Selected policies on train (in-sample for tuning)



| policy | net_SR | SR_lo95 | SR_hi95 | gross_SR | net_ret_pct | gross_ret_pct | vol_pct | cost_drag_pct | impact_pct | funding_pct | turnover_x_per_yr | gross_lev | n_bars | n_eff | n_blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 1.003 | -0.372 | 2.237 | 7.301 | 8.876 | 64.385 | 8.846 | 58.149 | 20.868 | -2.640 | 573.562 | 0.381 | 21936 | 21720.250 | 130 |
| EMA | 5.691 | 4.359 | 7.007 | 6.784 | 44.226 | 52.711 | 7.772 | 10.749 | 0.901 | -2.265 | 151.503 | 0.332 | 21936 | 21097.013 | 130 |
| Garleanu-Pedersen | 5.964 | 4.657 | 7.326 | 6.740 | 33.282 | 37.609 | 5.581 | 5.850 | 0.281 | -1.523 | 85.678 | 0.232 | 21936 | 20989.335 | 130 |
| No-trade band | 5.605 | 4.190 | 6.959 | 6.102 | 21.396 | 23.294 | 3.817 | 2.788 | 0.708 | -0.889 | 32.012 | 0.153 | 21936 | 20831.538 | 130 |
