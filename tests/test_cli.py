"""Smoke tests for the CLI."""
import importlib.util

import pytest
from click.testing import CliRunner

from matchify.cli import cli

_HAS_DEEP = importlib.util.find_spec("sentence_transformers") is not None


def test_cli_default_run(tmp_path):
    out = tmp_path / "out.html"
    result = CliRunner().invoke(
        cli,
        ["model-comparisons", "--limit", "30", "--no-confusion", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0


def test_cli_explicit_models_and_datasets(tmp_path):
    out = tmp_path / "out.html"
    result = CliRunner().invoke(
        cli,
        [
            "model-comparisons",
            "--dataset", "dblp-acm",
            "--dataset", "synthetic-people",
            "--models", "exact",
            "--models", "flex",
            "--limit", "30",
            "--no-confusion",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DBLP-ACM" in result.output
    assert "Synthetic People" in result.output


@pytest.mark.skipif(not _HAS_DEEP, reason="requires the [deep] extra")
def test_cli_all_flag(tmp_path):
    out = tmp_path / "out.html"
    result = CliRunner().invoke(
        cli,
        ["model-comparisons", "--all", "--limit", "30", "--no-confusion", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    # every bundled dataset should show up in the output
    assert "Amazon-Google Products" in result.output
    assert "DBLP-ACM" in result.output
    assert "Synthetic People" in result.output


def test_cli_unknown_dataset_errors(tmp_path):
    result = CliRunner().invoke(
        cli,
        ["model-comparisons", "--dataset", "not-a-real-dataset", "--limit", "10"],
    )
    assert result.exit_code != 0
    assert "Unknown dataset" in result.output
