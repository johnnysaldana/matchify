"""
Tests for threshold_sweep and the PR-curve plotting helper. Smoke level:
the sweep returns sane shapes, monotonic-ish stats, and the plot writes
a non-empty PNG.
"""
import os

import pytest


def test_threshold_sweep_shape(amazon_google_sample, amazon_google_field_config,
                               amazon_google_blocking_config, amazon_google_ignored):
    from matchify.models.flex_match_model import FlexMatchModel
    df = amazon_google_sample.copy()
    model = FlexMatchModel(
        df,
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
    )
    model.train()
    sweep = model.threshold_sweep([0.0, 0.25, 0.5, 0.75, 1.0])
    assert list(sweep.columns) == ['threshold', 'tp', 'fp', 'tn', 'fn',
                                   'precision', 'recall', 'f1']
    assert len(sweep) == 5
    # recall is monotonically non-increasing as threshold rises
    recalls = sweep.sort_values('threshold')['recall'].tolist()
    for a, b in zip(recalls, recalls[1:]):
        assert b <= a + 1e-9


def test_threshold_sweep_default_grid(amazon_google_sample, amazon_google_field_config,
                                      amazon_google_blocking_config, amazon_google_ignored):
    from matchify.models.flex_match_model import FlexMatchModel
    df = amazon_google_sample.copy()
    model = FlexMatchModel(
        df,
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
    )
    model.train()
    sweep = model.threshold_sweep()
    # default grid is 0..1 in 0.02 steps -> 51 rows
    assert len(sweep) == 51
    assert sweep['threshold'].iloc[0] == 0.0
    assert sweep['threshold'].iloc[-1] == 1.0


def test_save_pr_curve(tmp_path, amazon_google_sample, amazon_google_field_config,
                       amazon_google_blocking_config, amazon_google_ignored):
    from matchify.models.flex_match_model import FlexMatchModel
    from matchify.plotting import save_pr_curve

    df = amazon_google_sample.copy()
    model = FlexMatchModel(
        df,
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
    )
    model.train()
    sweep = model.threshold_sweep([0.0, 0.5, 1.0])
    out = tmp_path / "pr.png"
    save_pr_curve([("FlexMatchModel", sweep)], str(out), title="test")
    assert out.exists()
    assert out.stat().st_size > 0
