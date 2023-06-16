"""smoke tests for threshold_sweep + the PR curve helper."""


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


def test_threshold_sweep_default_uses_observed_scores(
    amazon_google_sample, amazon_google_field_config,
    amazon_google_blocking_config, amazon_google_ignored,
):
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
    # the default sweep tracks the observed score distribution rather
    # than a fixed grid. it should at minimum contain a recall=1 anchor
    # (threshold below every score) and a recall=0 anchor (threshold
    # above every score), with strictly non-increasing recall as
    # threshold rises.
    assert sweep['threshold'].iloc[0] == 0.0
    assert sweep['recall'].iloc[0] == 1.0
    assert sweep['recall'].iloc[-1] == 0.0
    recalls = sweep.sort_values('threshold')['recall'].tolist()
    for a, b in zip(recalls, recalls[1:]):
        assert b <= a + 1e-9


def test_threshold_sweep_handles_binary_scorer(
    amazon_google_sample, amazon_google_ignored,
):
    """ExactMatchModel emits 0/1 scores. The default sweep should still
    produce a recall=1 anchor, a real operating point, and a recall=0
    anchor without collapsing to a single row."""
    from matchify.models.exact_match_model import ExactMatchModel
    model = ExactMatchModel(amazon_google_sample.copy(), ignored_columns=amazon_google_ignored)
    sweep = model.threshold_sweep()
    assert sweep['recall'].iloc[0] == 1.0
    assert sweep['recall'].iloc[-1] == 0.0
    # at most a handful of distinct (recall, precision) points - exact
    # match is binary so anything beyond ~3 would mean we are
    # interpolating phantom operating points.
    assert sweep[['recall', 'precision']].drop_duplicates().shape[0] <= 4


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
