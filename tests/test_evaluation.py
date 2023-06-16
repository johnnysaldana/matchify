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


def test_aggregate_sweeps_interpolates_to_common_recall_grid():
    # two sweeps with different thresholds (mimicking observed-score
    # sweeps). the aggregation should align them on a common recall
    # axis and average precision per recall.
    a = pd.DataFrame({
        'threshold': [0.0, 0.4, 0.7, 1.0],
        'precision': [0.10, 0.30, 0.60, 1.00],
        'recall':    [1.00, 0.80, 0.40, 0.00],
        'f1':        [0.18, 0.44, 0.48, 0.00],
    })
    b = pd.DataFrame({
        'threshold': [0.0, 0.3, 0.6, 1.0],
        'precision': [0.20, 0.40, 0.70, 1.00],
        'recall':    [1.00, 0.90, 0.50, 0.00],
        'f1':        [0.33, 0.55, 0.58, 0.00],
    })
    out = aggregate_sweeps([a, b])
    assert 'recall' in out.columns and 'precision' in out.columns
    # endpoints are the trivial anchors
    assert out['recall'].iloc[0] == 0.0
    assert out['recall'].iloc[-1] == 1.0
    # at recall=1 both curves report precision=0.1 and 0.2 -> mean 0.15
    p_at_one = out[out['recall'] == 1.0]['precision'].iloc[0]
    assert abs(p_at_one - 0.15) < 1e-6
