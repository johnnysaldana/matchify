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
    Average a list of threshold_sweep DataFrames into one PR curve.

    Per-seed sweeps now use observed scores as thresholds, so the
    threshold sets differ across seeds and per-threshold averaging is
    no longer well-defined. Instead, interpolate each sweep's precision
    onto a common recall grid and average. F1 is recomputed from the
    averaged precision/recall so it stays consistent.
    """
    import numpy as np

    if not sweep_dfs:
        return pd.DataFrame()
    if len(sweep_dfs) == 1:
        return sweep_dfs[0].copy()

    grid = np.linspace(0.0, 1.0, 101)
    p_curves = []
    for s in sweep_dfs:
        # drop the trivial (recall=0, precision=0) sentinel that the
        # threshold_sweep emits at the highest threshold. keeping it
        # makes np.interp draw a phantom diagonal from the origin to
        # the leftmost real operating point, which appears on small
        # datasets where the smallest non-zero recall is well above 0.
        clean = s[~((s['recall'] == 0.0) & (s['precision'] == 0.0))]
        # for each recall, keep the best precision (Pareto frontier).
        # then sort by recall ascending so np.interp gets a monotone xp.
        agg = clean.groupby('recall', as_index=False)['precision'].max()
        agg = agg.sort_values('recall')
        r = agg['recall'].to_numpy(dtype=float)
        p = agg['precision'].to_numpy(dtype=float)
        # left=first_p extends precision below the smallest observed
        # recall as a flat plateau (sklearn convention: precision is
        # undefined when no predictions are made, so we anchor it to
        # the first real value rather than zero).
        # right=last_p extends to recall=1 cleanly.
        p_curves.append(np.interp(grid, r, p, left=float(p[0]), right=float(p[-1])))

    p_mean = np.mean(p_curves, axis=0)
    f1 = np.where(p_mean + grid > 0, 2 * p_mean * grid / (p_mean + grid), 0.0)
    return pd.DataFrame({'recall': grid, 'precision': p_mean, 'f1': f1})
