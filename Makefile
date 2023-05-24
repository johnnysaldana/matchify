.PHONY: install install-deep bench bench-quick test lint format clean help

help:
	@echo "matchify make targets:"
	@echo "  install        Editable install, base deps only (no torch)"
	@echo "  install-deep   Editable install + the [deep] extra (BERT and Siamese)"
	@echo "  bench          Run every model on every bundled dataset (500 rows each), write PR curves"
	@echo "  bench-quick    Same as 'bench' but on 100 rows. Sanity check, ~1 min"
	@echo "  test           pytest tests/"
	@echo "  lint           ruff check matchify/ tests/"
	@echo "  format         ruff format matchify/ tests/"
	@echo "  clean          Remove output.html and build artifacts"

install:
	pip install -e .

install-deep:
	pip install -e ".[deep]"

bench:
	matchify model-comparisons --all --limit 500 --pr-curves docs/pr/

bench-quick:
	matchify model-comparisons --all --limit 100

test:
	pytest tests/

lint:
	ruff check matchify/ tests/

format:
	ruff format matchify/ tests/

clean:
	rm -f output.html
	rm -rf build/ dist/ *.egg-info/ matchify.egg-info/
