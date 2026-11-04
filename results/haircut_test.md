## Selection-bias haircut, test

noise_max_SR = expected max Sharpe of n_trials pure-noise strategies; deflated_SR_prob = P(true SR > that). Read the no-trade band's Sharpe with its gross_lev (average gross exposure / capital), turnover and time in market (time_in_market = share of hours with any position; time_above_1pct = share with gross exposure >= 1% of capital): a Sharpe on a near-empty book is not comparable to the other policies.

| policy | net_SR | gross_lev | time_in_market | time_above_1pct | noise_max_SR | n_trials | prob_SR_above_0 | deflated_SR_prob |
|---|---|---|---|---|---|---|---|---|
| Naive | -53.572 | 0.190 | 1.000 | 1.000 | 2.059 | 94 | 0.000 | 0.000 |
| EMA | 1.479 | 0.069 | 1.000 | 1.000 | 2.057 | 94 | 0.965 | 0.240 |
| Garleanu-Pedersen | -0.162 | 0.062 | 1.000 | 1.000 | 2.087 | 94 | 0.421 | 0.003 |
| No-trade band | 0.506 | 0.003 | 1.000 | 0.000 | 2.049 | 94 | 0.731 | 0.030 |
