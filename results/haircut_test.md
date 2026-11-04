## Selection-bias haircut, test

noise_max_SR = expected max Sharpe of n_trials pure-noise strategies; deflated_SR_prob = P(true SR > that).

| policy | net_SR | noise_max_SR | n_trials | prob_SR_above_0 | deflated_SR_prob |
|---|---|---|---|---|---|
| Naive | -53.572 | 1.988 | 74 | 0.000 | 0.000 |
| EMA | 1.479 | 1.986 | 74 | 0.965 | 0.268 |
| Garleanu-Pedersen | -0.162 | 2.015 | 74 | 0.421 | 0.004 |
| No-trade band | 0.506 | 1.979 | 74 | 0.731 | 0.036 |
