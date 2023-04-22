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
common abstract `ERBaseModel` interface and ships three concrete models
spanning the methodological progression in the field:

- `ExactMatchModel` — a hash-based exact-match baseline. No training, no
  blocking, no field-aware logic. Useful as a lower bound and as a sanity
  check on a deduplicated dataset.
- `FlexMatchModel` — a configurable, field-aware similarity model. Supports
  per-field type-aware normalization (`name`, `phone`, `address`, `date`),
  per-field comparison methods (Jaro-Winkler, Levenshtein, TF-IDF cosine,
  Jaccard), and blocking strategies (`prefix`, `sorted_neighborhood`,
  `block`, `full`). The TF-IDF vectorizer is fit on the corpus during
  `train()`.
- `MLPMatchModel` — a supervised multilayer perceptron over a fixed
  feature vector of per-field similarity scores (Jaro-Winkler, Levenshtein,
  TF-IDF cosine, Jaccard). Training pairs are sampled 50/50 from the
  `group_id` supervision; the model learns which features matter on each
  dataset rather than relying on a hand-tuned weighting.

Every model implements `mrr()` and `confusion_matrix(threshold)` on the base
class so they can be compared apples-to-apples on any labelled benchmark.

A small data-splitting utility (`DataSplitter`), a set of data loaders (CSV,
JSON, Parquet, S3, SQL), a synthetic person-record generator, and a CLI for
generating side-by-side model comparison tables round out the package.

## Results

MRR and confusion-matrix figures on the first 500 records of each bundled
benchmark, scored on every other record, with `--threshold 0.5` for the
confusion matrix:

### Amazon-Google Products (500 records)

| Model | MRR | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ExactMatchModel | 0.280 | 1.000 | 0.215 | 0.354 |
| FlexMatchModel | 0.489 | 0.101 | 0.842 | 0.180 |
| MLPMatchModel | **0.576** | 0.229 | **1.000** | **0.372** |

### DBLP-ACM (500 records)

| Model | MRR | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ExactMatchModel | 0.000 | 0.000 | 0.000 | 0.000 |
| FlexMatchModel | 0.924 | 0.313 | 1.000 | 0.477 |
| MLPMatchModel | **0.924** | **1.000** | **1.000** | **1.000** |

The MLP wins overall: it inherits the recall of FlexMatchModel (its features
include all of FlexMatchModel's similarity metrics) but learns a calibrated
match boundary that drops the precision-shredding false positives. On DBLP-ACM
the field signal is nearly deterministic so the MLP achieves perfect
classification at threshold 0.5.

Reproduce with:

```bash
matchify model-comparisons \
  --dataset amazon-google --dataset dblp-acm \
  --models exact --models flex --models mlp \
  --limit 500
```

## Installation

```bash
git clone https://github.com/johnnysaldana/matchify.git
cd matchify
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10–3.12 is supported. `gensim` does not yet build on 3.13+; pin to
3.12 if you hit a `gensim` install error.

## Quickstart

```python
import pandas as pd
from matchify import ExactMatchModel, FlexMatchModel
from matchify.models.mlp_match_model import MLPMatchModel

df = pd.read_csv('datasets/Amazon-GoogleProducts/CombinedProducts.csv')
ignored = ['id', 'group_id', 'original_id']

field_config = {
    'name':         {'type': 'other', 'comparison_method': 'jaro_winkler'},
    'description':  {'type': 'other', 'comparison_method': 'tfidf_cosine'},
    'manufacturer': {'type': 'other', 'comparison_method': 'jaro_winkler'},
    'price':        {'type': 'other', 'comparison_method': 'jaro_winkler'},
}
blocking_config = {'name': {'method': 'prefix', 'threshold': 2}}

emm = ExactMatchModel(df, ignored_columns=ignored)
print('Exact-match MRR:', emm.mrr())

fmm = FlexMatchModel(df, field_config=field_config,
                     blocking_config=blocking_config, ignored_columns=ignored)
fmm.train()
print('Flex-match MRR:', fmm.mrr())

mlp = MLPMatchModel(df, field_config=field_config,
                    blocking_config=blocking_config, ignored_columns=ignored)
mlp.train()
print('MLP MRR:', mlp.mrr())
print('MLP confusion @0.5:', mlp.confusion_matrix(threshold=0.5))

# Predict matches for a single lookup record
record = df.iloc[7].drop(labels=ignored)
print(mlp.predict(record).head(10))
```

To regenerate the model-comparison HTML report, run from the project root:

```bash
matchify model-comparisons --dataset amazon-google --dataset dblp-acm \
                           --models exact --models flex --models mlp
```

The output is written to `output.html`.

## Datasets

Two benchmarks from the [Leipzig DB-Group benchmark
collection](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution)
ship with the repo. Each follows the same `Combined*.csv` layout: one row
per record, with an `id`, the original source-table identifier under
`original_id`, and a `group_id` for supervision (records with the same
`group_id` are duplicates of one another).

- `datasets/Amazon-GoogleProducts/` — e-commerce, ~3.7K records,
  fields: name, description, manufacturer, price.
- `datasets/DBLP-ACM/` — bibliographic, ~5K records,
  fields: title, authors, venue, year.

Each directory has a small notebook (`Create*Dataset.ipynb`) documenting how
the combined CSV was built from the upstream source tables.

## Repository layout

```
matchify/
├── matchify/
│   ├── models/
│   │   ├── base_model.py        # ERBaseModel + MRR + confusion_matrix
│   │   ├── exact_match_model.py
│   │   ├── flex_match_model.py
│   │   └── mlp_match_model.py
│   ├── pipelines/entity_resolution_pipeline.py
│   ├── utils/
│   │   ├── data_splitter.py
│   │   ├── data_loaders/        # csv, json, parquet, s3, sql
│   │   └── synthetic_data_generation/person_faker.py
│   ├── templates/table.html     # CLI report template
│   ├── datasets.py              # bundled-benchmark registry
│   └── cli.py
├── datasets/
│   ├── Amazon-GoogleProducts/
│   └── DBLP-ACM/
├── examples/                    # quickstart notebooks
├── tests/                       # pytest smoke tests
├── run.ipynb                    # original research notebook
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
