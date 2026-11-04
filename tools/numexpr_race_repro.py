"""Minimal reproduction of the numexpr multithreading bug behind the intermittent NaNs.

    python tools/numexpr_race_repro.py [threads] [iterations]

Evaluates "a & b" on fixed random boolean arrays (40000 x 192) and compares each result with
numpy. On the author's machine (Windows 11, 14 logical cores, numexpr 2.14.1, numpy 2.3.5,
Python 3.13.9) about 0.2-0.3% of multithreaded calls returned a wrong (all-False) result, and
none did with 1 thread. Not part of the test suite: the failure is rare and machine-dependent.
"""
import sys

import numexpr as ne
import numpy as np

threads = int(sys.argv[1]) if len(sys.argv) > 1 else ne.detect_number_of_cores()
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
ne.set_num_threads(threads)
rng = np.random.default_rng(0)
a = rng.random((40000, 192)) < 0.6
b = rng.random((40000, 192)) < 0.9
ref = a & b
wrong = sum(int((ne.evaluate("a & b") != ref).any()) for _ in range(n_iter))
print(f"numexpr {ne.__version__}, {threads} threads: {wrong}/{n_iter} wrong results")
