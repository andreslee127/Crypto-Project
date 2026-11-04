## Post-hoc: break-even cost of the stand-alone 4h reversal

POST-HOC, NOT A PRE-REGISTERED RESULT. Train and test windows (no holdout). Stand-alone rev_4h portfolio (weights ~ signal / vol, gross leverage 1), before any trading policy. Break-even = gross return / turnover: the one-way cost per unit traded at which net return is 0. A negative value means the gross return was already negative.

| window | gross_ret_pct_per_yr | turnover_x_per_yr | breakeven_cost_bps |
|---|---|---|---|
| train | 45.930 | 6161.176 | 0.745 |
| test | -18.537 | 5998.912 | -0.309 |
