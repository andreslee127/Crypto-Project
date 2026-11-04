## Grid-edge check (added after the first real run)

Train decides; the test columns are POST-HOC and do not change any selection.

| policy | param | train_net_SR | test_net_SR_posthoc | train_turnover |
|---|---|---|---|---|
| ema | 0.002 | -0.211 | 1.622 | 7.057 |
| ema | 0.001 | -0.090 | 1.197 | 3.629 |
| gp | 32.000 | 0.276 | 0.003 | 12.936 |
| gp | 64.000 | 0.317 | 0.170 | 8.741 |
| gp | 128.000 | 0.324 | 0.366 | 5.750 |
