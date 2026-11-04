## SYNTHETIC DATA - pipeline check only. Selection-bias haircut, test

noise_max_SR = expected max Sharpe of n_trials pure-noise strategies; deflated_SR_prob = P(true SR > that).

| policy | net_SR | noise_max_SR | n_trials | prob_SR_above_0 | deflated_SR_prob |
|---|---|---|---|---|---|
| Naive | 2.998 | 2.789 | 69 | 1.000 | 0.601 |
| EMA | 6.828 | 2.789 | 69 | 1.000 | 1.000 |
| Garleanu-Pedersen | 7.097 | 2.789 | 69 | 1.000 | 1.000 |
| No-trade band | 6.741 | 2.789 | 69 | 1.000 | 1.000 |
