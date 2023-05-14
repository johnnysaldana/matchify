"""
Tests for the type-aware normalizers on ERBaseModel.

Both bundled real-world benchmarks (Amazon-Google, DBLP-ACM) use
type='other' and bypass these code paths entirely; only the
synthetic-people benchmark exercises them, so these tests guard
against regressions in the rarely-traveled normalization paths.
"""
import pandas as pd
import pytest

from matchify.models.base_model import ERBaseModel


class _StubModel(ERBaseModel):
    """Concrete subclass to satisfy ERBaseModel's ABC contract."""

    def preprocess(self, df, ignored_columns=None):
        return df

    def train(self, *args, **kwargs):
        pass

    def predict(self, record, **kwargs):
        return pd.DataFrame()


@pytest.fixture
def model():
    return _StubModel(pd.DataFrame())


def test_normalize_name_lowercases_and_parses(model):
    out = model._normalize_name("John A. Smith")
    assert out == out.lower()
    assert "smith" in out


def test_normalize_name_handles_complex_input(model):
    # Should not raise on names with apostrophes, hyphens, suffixes
    assert "o'neil" in model._normalize_name("Mary-Jane O'Neil III")


def test_normalize_phone_us_default(model):
    # US-formatted numbers without explicit country prefix should parse
    assert model._normalize_phone("(415) 555-1234") == "+14155551234"
    assert model._normalize_phone("415.555.1234") == "+14155551234"


def test_normalize_phone_international(model):
    assert model._normalize_phone("+44 20 7946 0958").startswith("+44")


def test_normalize_phone_falls_back_to_digits(model):
    # If phonenumbers can't make sense of the string at all, we shouldn't
    # crash - we return an empty string.
    assert model._normalize_phone("garbage") == ""
    # Empty input is benign.
    assert model._normalize_phone("") == ""


def test_normalize_address_lowercases(model):
    out = model._normalize_address("123 Main St, Springfield, IL 62701")
    assert out == out.lower()
    assert "springfield" in out


def test_normalize_date_iso(model):
    assert model._normalize_date("1959-10-13") == "1959-10-13"
    assert model._normalize_date("Mar 5, 1990") == "1990-03-05"
    assert model._normalize_date("not-a-date") == ""
