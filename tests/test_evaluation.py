"""Tests for the multi-seed aggregation helpers."""
import pandas as pd

from matchify.evaluation import aggregate_metric, aggregate_sweeps, is_stochastic


def test_is_stochastic():
    assert is_stochastic("mlp")
    assert is_stochastic("siamese")
    assert not is_stochastic("exact")
    assert not is_stochastic("flex")
    assert not is_stochastic("bert")


def test_aggregate_metric_single():
    mean, std = aggregate_metric([0.5])
    assert mean == 0.5
    assert std == 0.0


def test_aggregate_metric_multi():
    mean, std = aggregate_metric([0.4, 0.5, 0.6])
    assert abs(mean - 0.5) < 1e-9
    assert std > 0


def test_aggregate_metric_empty():
    mean, std = aggregate_metric([])
    assert mean == 0.0
    assert std == 0.0


def test_aggregate_sweeps_averages_per_threshold():
    a = pd.DataFrame({
        'threshold': [0.0, 0.5, 1.0],
        'precision': [0.1, 0.5, 0.9],
        'recall':    [0.9, 0.5, 0.1],
        'f1':        [0.18, 0.5, 0.18],
        'tp': [10, 5, 1], 'fp': [90, 50, 10], 'tn': [10, 50, 90], 'fn': [1, 5, 10],
    })
    b = pd.DataFrame({
        'threshold': [0.0, 0.5, 1.0],
        'precision': [0.2, 0.6, 1.0],
        'recall':    [1.0, 0.6, 0.2],
        'f1':        [0.33, 0.6, 0.33],
        'tp': [12, 6, 2], 'fp': [88, 48, 8], 'tn': [12, 52, 92], 'fn': [0, 4, 8],
    })
    out = aggregate_sweeps([a, b])
    assert len(out) == 3
    p_at_0_5 = out[out['threshold'] == 0.5]['precision'].iloc[0]
    assert abs(p_at_0_5 - 0.55) < 1e-9
