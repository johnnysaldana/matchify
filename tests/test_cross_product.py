"""
Cross-product smoke tests. Every bundled (dataset, model) combo on a
tiny slice. Catches integration bugs that per-model tests miss, like
a dataset's field_config not playing well with a model's preprocessing,
or a registry entry pointing at a stale CSV.

Just trains and predicts a single record per combo. No MRR or confusion
matrix here, those run in test_models.py.
"""
import importlib.util
import warnings

import pandas as pd
import pytest

from matchify.datasets import DATASETS

warnings.filterwarnings("ignore")


_HAS_DEEP = importlib.util.find_spec("sentence_transformers") is not None


def _build(model_name, df, ignored, fc, bc):
    df = df.copy()
    if model_name == "exact":
        from matchify.models.exact_match_model import ExactMatchModel
        return ExactMatchModel(df, ignored_columns=ignored)
    if model_name == "flex":
        from matchify.models.flex_match_model import FlexMatchModel
        return FlexMatchModel(df, field_config=fc, blocking_config=bc, ignored_columns=ignored)
    if model_name == "mlp":
        from matchify.models.mlp_match_model import MLPMatchModel
        return MLPMatchModel(
            df, field_config=fc, blocking_config=bc, ignored_columns=ignored,
            n_pairs=200, max_iter=80,
        )
    if model_name == "bert":
        from matchify.models.bert_match_model import BertMatchModel
        return BertMatchModel(df, field_config=fc, blocking_config=bc, ignored_columns=ignored)
    if model_name == "siamese":
        from matchify.models.siamese_match_model import SiameseMatchModel
        return SiameseMatchModel(
            df, field_config=fc, blocking_config=bc, ignored_columns=ignored,
            n_pairs=200, epochs=1, batch_size=16,
        )
    raise ValueError(model_name)


def _id(combo):
    return f"{combo[0]}__{combo[1]}"


_BASIC_MODELS = ["exact", "flex", "mlp"]
_DEEP_MODELS = ["bert", "siamese"]
_DATASETS = list(DATASETS.keys())

_BASIC_COMBOS = [(m, d) for m in _BASIC_MODELS for d in _DATASETS]
_DEEP_COMBOS = [(m, d) for m in _DEEP_MODELS for d in _DATASETS]


@pytest.mark.parametrize("model_name,dataset_key", _BASIC_COMBOS, ids=_id)
def test_basic_model_on_dataset(model_name, dataset_key):
    cfg = DATASETS[dataset_key]
    df = pd.read_csv(cfg["path"]).head(40)
    model = _build(
        model_name,
        df,
        cfg["ignored_columns"],
        cfg["field_config"],
        cfg["blocking_config"],
    )
    if hasattr(model, "train"):
        model.train()
    record = df.iloc[0].drop(labels=cfg["ignored_columns"])
    preds = model.predict(record, only_matches=False, return_full_record=True)
    assert preds is not None
    assert "score" in preds.columns


@pytest.mark.skipif(not _HAS_DEEP, reason="requires the [deep] extra")
@pytest.mark.parametrize("model_name,dataset_key", _DEEP_COMBOS, ids=_id)
def test_deep_model_on_dataset(model_name, dataset_key):
    cfg = DATASETS[dataset_key]
    df = pd.read_csv(cfg["path"]).head(40)
    model = _build(
        model_name,
        df,
        cfg["ignored_columns"],
        cfg["field_config"],
        cfg["blocking_config"],
    )
    model.train()
    record = df.iloc[0].drop(labels=cfg["ignored_columns"])
    preds = model.predict(record, only_matches=False, return_full_record=True)
    assert preds is not None
    assert "score" in preds.columns


def test_dataset_registry_files_exist():
    """Every registry entry points at a readable CSV with the supervision
    columns the rest of the code expects."""
    import os
    for key, cfg in DATASETS.items():
        assert os.path.isfile(cfg["path"]), f"{key}: missing CSV at {cfg['path']}"
        df = pd.read_csv(cfg["path"], nrows=5)
        assert "id" in df.columns, f"{key}: no 'id' column"
        assert "group_id" in df.columns, f"{key}: no 'group_id' column"
        for field in cfg["field_config"]:
            assert field in df.columns, f"{key}: declared field '{field}' missing from CSV"
