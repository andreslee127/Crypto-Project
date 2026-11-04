"""From signal to portfolio: short-horizon alpha and optimal trading under transaction costs."""
import pandas as pd

# Precaution: during development, identical computations intermittently returned NaN in some
# processes on the author's machine. The cause was not isolated; pandas' multithreaded numexpr
# and bottleneck paths were the main suspects. With them off, pandas uses plain numpy, and two
# full pipeline runs give identical outputs (see report.md, reproducibility).
pd.set_option("compute.use_bottleneck", False)
pd.set_option("compute.use_numexpr", False)
