# matchify

A Python package for entity resolution (record linkage / deduplication) that
implements representative methods from major areas of the literature, evaluated
with mean reciprocal rank on standard benchmark datasets.

This repository is the artifact of an independent research project carried out
at Johns Hopkins University in Spring 2023 (course `EN.601.507: Applied Entity
Resolution & Deduplication`, advised by Dr. Tom Lippincott). It is archived on
Zenodo for citation; see `CITATION.cff` for the canonical citation.

## Overview

Entity resolution is the task of identifying records — across one or more data
sources — that refer to the same real-world entity. `matchify` exposes a
common abstract `ERBaseModel` interface and ships two concrete implementations
spanning the methodological spectrum used as baselines and middle-ground
references in the field:

- `ExactMatchModel` — a hash-based exact-match baseline. No training, no
  blocking, no field-aware logic. Useful as a lower bound and as a sanity
  check on a deduplicated dataset.
- `FlexMatchModel` — a configurable, field-aware similarity model. Supports
  per-field type-aware normalization (`name`, `phone`, `address`, `date`),
  per-field comparison methods (Jaro-Winkler, Levenshtein, TF-IDF cosine,
  Jaccard), and blocking strategies (`prefix`, `sorted_neighborhood`,
  `block`, `full`). The TF-IDF vectorizer is fit on the corpus during
  `train()`.

Both models are evaluated with mean reciprocal rank (MRR) over every record in
a labeled dataset: for each record, the model returns a ranked candidate list
and we score the reciprocal rank of the first true match.

A small data-splitting utility (`DataSplitter`), a set of data loaders (CSV,
JSON, Parquet, S3, SQL), a synthetic person-record generator, and a CLI for
generating side-by-side model comparison tables round out the package.

## Installation

```bash
git clone https://github.com/johnnysaldana/matchify.git
cd matchify
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.8–3.12 is supported. `gensim` does not yet build on 3.13+; pin to
3.12 if you hit a `gensim` install error.

## Quickstart

```python
import pandas as pd
from matchify import ExactMatchModel, FlexMatchModel

df = pd.read_csv('datasets/Amazon-GoogleProducts/CombinedProducts.csv')
ignored = ['id', 'group_id', 'original_id']

# Baseline
emm = ExactMatchModel(df, ignored_columns=ignored)
print('Exact-match MRR:', emm.mrr())

# Field-aware similarity model
field_config = {
    'name':         {'type': 'other', 'comparison_method': 'jaro_winkler'},
    'description':  {'type': 'other', 'comparison_method': 'tfidf_cosine'},
    'manufacturer': {'type': 'other', 'comparison_method': 'jaro_winkler'},
    'price':        {'type': 'other', 'comparison_method': 'jaro_winkler'},
}
blocking_config = {'name': {'method': 'prefix', 'threshold': 2}}

fmm = FlexMatchModel(
    df,
    field_config=field_config,
    blocking_config=blocking_config,
    ignored_columns=ignored,
)
fmm.train()
print('Flex-match MRR:', fmm.mrr())

# Predict matches for a single lookup record
record = df.iloc[7].drop(labels=ignored)
print(fmm.predict(record).head(10))
```

To regenerate the model-comparison HTML report bundled with the repo, run from
the project root:

```bash
matchify model-comparisons
# or, equivalently:
python -m matchify.cli model-comparisons
```

The output is written to `output.html`.

## Datasets

The Amazon-Google Products benchmark from the
[Leipzig DB-Group benchmarks](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution)
is included under `datasets/Amazon-GoogleProducts/`:

- `AmazonProducts.csv`, `GoogleProducts.csv` — the two source catalogs.
- `perfectmappings.csv` — the gold-standard alignment between them.
- `CombinedProducts.csv` — the two catalogs concatenated and joined to the
  gold-standard mapping; this is the input to the models. The
  `CreateProductsDataset.ipynb` notebook in that directory documents how it
  was built.

The `group_id` column is the supervision signal used by `mrr()`.

## Repository layout

```
matchify/
├── matchify/
│   ├── models/
│   │   ├── base_model.py        # ERBaseModel + MRR evaluation
│   │   ├── exact_match_model.py
│   │   └── flex_match_model.py
│   ├── pipelines/
│   │   └── entity_resolution_pipeline.py
│   ├── utils/
│   │   ├── data_splitter.py
│   │   ├── data_loaders/        # csv, json, parquet, s3, sql
│   │   └── synthetic_data_generation/person_faker.py
│   ├── templates/table.html     # CLI report template
│   └── cli.py
├── datasets/Amazon-GoogleProducts/
├── run.ipynb                    # research notebook
├── CITATION.cff
├── .zenodo.json
├── PUBLISHING.md                # Zenodo release checklist
└── setup.py
```

## Citation

If you use this software, please cite it. The canonical citation lives in
`CITATION.cff` and on the Zenodo record (DOI to be added on first release).

## License

MIT — see `LICENSE`.

## Acknowledgements

Advised by Dr. Tom Lippincott (Center for Language and Speech Processing,
Johns Hopkins University). The literature reading list, dataset survey, and
methodological framing draw on Christen's *Data Matching* (2012), Papadakis et
al.'s *The Four Generations of Entity Resolution* (2021), and the references
collected during the course; see `Entity Resolution Project EN.601.507/`
(local-only, not distributed) for the working bibliography.
