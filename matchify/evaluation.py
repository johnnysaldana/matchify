from statistics import mean, stdev

import pandas as pd

_STOCHASTIC = {"mlp", "siamese"}


def is_stochastic(model_name: str) -> bool:
    return model_name in _STOCHASTIC


def aggregate_metric(values):
    """Return (mean, stddev) for a list of floats. stddev=0 for n<2."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def aggregate_sweeps(sweep_dfs) -> pd.DataFrame:
    """
    Average a list of threshold_sweep DataFrames into one.

    Each sweep is indexed by threshold. The mean over precision/recall/F1
    per threshold is the curve we plot. Threshold count is fixed (51 by
    default) so simple per-position mean works.
    """
    if not sweep_dfs:
        return pd.DataFrame()
    if len(sweep_dfs) == 1:
        return sweep_dfs[0]
    stacked = pd.concat(sweep_dfs)
    return stacked.groupby('threshold', as_index=False).mean(numeric_only=True)
