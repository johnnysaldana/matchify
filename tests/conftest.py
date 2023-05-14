import os

import pandas as pd
import pytest

from matchify.datasets import DATASETS, REPO_ROOT


@pytest.fixture(scope="session")
def amazon_google_sample():
    """Small slice of the Amazon-Google benchmark for fast tests."""
    path = DATASETS["amazon-google"]["path"]
    return pd.read_csv(path).head(100)


@pytest.fixture(scope="session")
def dblp_acm_sample():
    """Small slice of the DBLP-ACM benchmark for fast tests."""
    path = DATASETS["dblp-acm"]["path"]
    return pd.read_csv(path).head(100)


@pytest.fixture(scope="session")
def synthetic_people_sample():
    """Small slice of the SyntheticPeople benchmark for fast tests."""
    path = DATASETS["synthetic-people"]["path"]
    return pd.read_csv(path).head(100)


@pytest.fixture
def synthetic_people_field_config():
    return DATASETS["synthetic-people"]["field_config"]


@pytest.fixture
def synthetic_people_blocking_config():
    return DATASETS["synthetic-people"]["blocking_config"]


@pytest.fixture
def synthetic_people_ignored():
    return DATASETS["synthetic-people"]["ignored_columns"]


@pytest.fixture
def amazon_google_field_config():
    return {
        "name":         {"type": "other", "comparison_method": "jaro_winkler"},
        "description":  {"type": "other", "comparison_method": "tfidf_cosine"},
        "manufacturer": {"type": "other", "comparison_method": "jaro_winkler"},
        "price":        {"type": "other", "comparison_method": "jaro_winkler"},
    }


@pytest.fixture
def amazon_google_blocking_config():
    return {"name": {"method": "prefix", "threshold": 2}}


@pytest.fixture
def amazon_google_ignored():
    return ["id", "group_id", "original_id"]
