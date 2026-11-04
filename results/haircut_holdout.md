## Selection-bias haircut, holdout

noise_max_SR = expected max Sharpe of n_trials pure-noise strategies; deflated_SR_prob = P(true SR > that).

| policy | net_SR | noise_max_SR | n_trials | prob_SR_above_0 | deflated_SR_prob |
|---|---|---|---|---|---|
| Naive | -48.967 | 2.250 | 119 | 0.000 | 0.000 |
| EMA | -0.977 | 2.276 | 119 | 0.129 | 0.000 |
| Garleanu-Pedersen | 0.614 | 2.254 | 119 | 0.760 | 0.030 |
| No-trade band | 0.924 | 2.253 | 119 | 0.855 | 0.064 |
