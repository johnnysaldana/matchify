"""Smoke tests for the bundled models.

These run on a 100-row slice of Amazon-Google and assert that each model
produces a sane MRR (>= 0 and <= 1) and a non-empty prediction set. The
goal is to catch regressions in the public API and basic plumbing - not
to validate accuracy.
"""
import warnings

import pytest


warnings.filterwarnings("ignore")


def test_exact_match_model(amazon_google_sample, amazon_google_ignored):
    from matchify.models.exact_match_model import ExactMatchModel
    df = amazon_google_sample.copy()
    model = ExactMatchModel(df, ignored_columns=amazon_google_ignored)
    record = df.iloc[0].drop(labels=amazon_google_ignored)
    preds = model.predict(record, only_matches=False, return_full_record=True)
    assert not preds.empty
    mrr = model.mrr()
    assert 0.0 <= mrr <= 1.0


def test_flex_match_model(
    amazon_google_sample, amazon_google_ignored,
    amazon_google_field_config, amazon_google_blocking_config,
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
    record = df.iloc[0].drop(labels=amazon_google_ignored)
    preds = model.predict(record)
    assert not preds.empty
    mrr = model.mrr()
    assert 0.0 <= mrr <= 1.0


def test_mlp_match_model(
    amazon_google_sample, amazon_google_ignored,
    amazon_google_field_config, amazon_google_blocking_config,
):
    from matchify.models.mlp_match_model import MLPMatchModel
    df = amazon_google_sample.copy()
    model = MLPMatchModel(
        df,
        field_config=amazon_google_field_config,
        blocking_config=amazon_google_blocking_config,
        ignored_columns=amazon_google_ignored,
        n_pairs=200,
        max_iter=100,
    )
    model.train()
    record = df.iloc[0].drop(labels=amazon_google_ignored)
    preds = model.predict(record)
    assert not preds.empty
    mrr = model.mrr()
    assert 0.0 <= mrr <= 1.0


def test_flex_match_model_type_aware_normalization(
    synthetic_people_sample, synthetic_people_ignored,
    synthetic_people_field_config, synthetic_people_blocking_config,
):
    """
    The synthetic-people benchmark is the only bundled dataset that
    exercises the type-aware normalization paths in ERBaseModel
    (_normalize_name, _normalize_phone, _normalize_address,
    _normalize_date). This test guards against regressions there.
    """
    from matchify.models.flex_match_model import FlexMatchModel
    df = synthetic_people_sample.copy()
    model = FlexMatchModel(
        df,
        field_config=synthetic_people_field_config,
        blocking_config=synthetic_people_blocking_config,
        ignored_columns=synthetic_people_ignored,
    )
    model.train()
    record = df.iloc[0].drop(labels=synthetic_people_ignored)
    preds = model.predict(record)
    assert not preds.empty
    mrr = model.mrr()
    assert 0.0 <= mrr <= 1.0


def test_confusion_matrix_basic(
    amazon_google_sample, amazon_google_ignored,
    amazon_google_field_config, amazon_google_blocking_config,
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
    cm = model.confusion_matrix(threshold=0.5)
    for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1"):
        assert key in cm
    assert 0.0 <= cm["precision"] <= 1.0
    assert 0.0 <= cm["recall"] <= 1.0


_HAS_DEEP = (
    __import__("importlib.util").util.find_spec("sentence_transformers") is not None
)


@pytest.mark.skipif(not _HAS_DEEP, reason="requires the [deep] extra")
def test_bert_match_model(
    dblp_acm_sample,
    amazon_google_ignored,
):
    from matchify.models.bert_match_model import BertMatchModel
    df = dblp_acm_sample.copy()
    field_config = {"title": {}, "authors": {}, "venue": {}, "year": {}}
    blocking_config = {"title": {"method": "prefix", "threshold": 3}}
    model = BertMatchModel(
        df,
        field_config=field_config,
        blocking_config=blocking_config,
        ignored_columns=amazon_google_ignored,
    )
    model.train()
    record = df.iloc[0].drop(labels=amazon_google_ignored)
    preds = model.predict(record)
    assert not preds.empty


@pytest.mark.skipif(not _HAS_DEEP, reason="requires the [deep] extra")
def test_siamese_match_model(
    dblp_acm_sample,
    amazon_google_ignored,
):
    from matchify.models.siamese_match_model import SiameseMatchModel
    df = dblp_acm_sample.copy()
    field_config = {"title": {}, "authors": {}, "venue": {}, "year": {}}
    blocking_config = {"title": {"method": "prefix", "threshold": 3}}
    model = SiameseMatchModel(
        df,
        field_config=field_config,
        blocking_config=blocking_config,
        ignored_columns=amazon_google_ignored,
        n_pairs=200,
        epochs=1,
        batch_size=16,
    )
    model.train()
    record = df.iloc[0].drop(labels=amazon_google_ignored)
    preds = model.predict(record)
    assert not preds.empty
