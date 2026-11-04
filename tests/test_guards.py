"""Importing s2p must switch off pandas' numexpr path (root cause of the intermittent NaN bug)."""
import pandas as pd

import s2p  # noqa: F401


def test_numexpr_and_bottleneck_disabled_in_pandas():
    assert pd.get_option("compute.use_numexpr") is False
    assert pd.get_option("compute.use_bottleneck") is False
