"""Tests for the held-out group split on ERBaseModel."""
import pandas as pd

from matchify.datasets import DATASETS


def test_split_back_compat(amazon_google_sample, amazon_google_field_config,
                           amazon_google_blocking_config, amazon_google_ignored):
    """test_size=0 means no split, everything is train."""
    from matchify.models.flex_match_model import FlexMatchModel
    df = amazon_google_sample.copy()
    model = FlexMatchModel(
        df,
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
    )
    assert len(model._train_idx) == len(df)
    assert len(model._test_idx) == 0
    assert len(model._eval_idx) == len(df)


def test_split_partitions_by_group(amazon_google_sample, amazon_google_field_config,
                                   amazon_google_blocking_config, amazon_google_ignored):
    """test_size>0, test rows all share group_ids that aren't in train."""
    from matchify.models.flex_match_model import FlexMatchModel
    df = amazon_google_sample.copy()
    model = FlexMatchModel(
        df,
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
        test_size=0.3,
        random_state=0,
    )
    assert len(model._test_idx) > 0
    # train + test cover the full df, no overlap
    train_set = set(model._train_idx)
    test_set = set(model._test_idx)
    assert train_set & test_set == set()
    assert len(train_set) + len(test_set) == len(df)
    # every test row's group_id is exclusively in the test partition
    test_groups = set(df.loc[model._test_idx, 'group_id'].dropna())
    train_groups = set(df.loc[model._train_idx, 'group_id'].dropna())
    assert test_groups & train_groups == set()


def test_split_is_deterministic_per_seed(amazon_google_sample, amazon_google_field_config,
                                         amazon_google_blocking_config, amazon_google_ignored):
    from matchify.models.flex_match_model import FlexMatchModel
    df = amazon_google_sample.copy()
    a = FlexMatchModel(
        df.copy(),
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
        test_size=0.3, random_state=42,
    )
    b = FlexMatchModel(
        df.copy(),
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
        test_size=0.3, random_state=42,
    )
    assert list(a._test_idx) == list(b._test_idx)


def test_supervised_pair_sampling_respects_train_partition():
    # Catches the bug where pair indices crossed into test rows.
    from matchify.models.mlp_match_model import MLPMatchModel
    cfg = DATASETS['amazon-google']
    df = pd.read_csv(cfg['path']).head(100)
    model = MLPMatchModel(
        df,
        field_config=cfg['field_config'],
        blocking_config=cfg['blocking_config'],
        ignored_columns=cfg['ignored_columns'],
        n_pairs=200, max_iter=50,
        test_size=0.3, random_state=0,
    )
    train_set = set(model._train_idx)
    positives, negatives = model._sample_training_pairs()
    for i, j in positives + negatives:
        assert i in train_set, f"pair {i} not in train partition"
        assert j in train_set, f"pair {j} not in train partition"
