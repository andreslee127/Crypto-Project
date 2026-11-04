"""From signal to portfolio: short-horizon alpha and optimal trading under transaction costs."""
import pandas as pd

# Root cause of the intermittent NaN / wrong-value bug (see report.md section 8 and
# tools/numexpr_race_repro.py): numexpr 2.14.1's MULTITHREADED evaluate() occasionally returns an
# output array that was never filled. Plain numexpr on fixed inputs: 3/1000 wrong results for
# "a & b" and 1/1000 for "a * b" at 40000 x 192, 4/2000 with 2 threads, and 0/2000 with 1 thread.
# pandas sends element-wise operations on frames above ~1e6 cells through numexpr, so a boolean
# mask or a product could silently come back all False / all zero. Disabling numexpr in pandas
# removes the exposure. bottleneck was tested separately (0/2000 differences); it stays off only
# so that results remain bit-identical to the published run.
pd.set_option("compute.use_numexpr", False)
pd.set_option("compute.use_bottleneck", False)
