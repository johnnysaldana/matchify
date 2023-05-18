# Abt-Buy

E-commerce entity-resolution benchmark from the [Leipzig DB-Group benchmark
collection](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution).
Two product catalogs (Abt and Buy.com) with a hand-curated mapping of the
records that refer to the same product.

`CombinedProducts.csv` is built by `CreateProductsDataset.ipynb` and follows
the same layout as the other bundled benchmarks: one row per record, with
`id` (unique), `original_id` (source-table id), and `group_id` (records
sharing a `group_id` are duplicates of one another).

Composition:
- 1081 Abt records, 1092 Buy records, 2173 total
- 1097 perfect-match pairs in the upstream mapping. Some records are
  matched many-to-one, so the union-find pass collapses those into 1076
  groups (1055 of size 2, 21 of size 3)

Smaller than Amazon-Google (~3.7K) and DBLP-ACM (~5K), so it's the
quickest of the real-world benchmarks to run end-to-end.

Fields: `name`, `description`, `manufacturer`, `price`. Abt has no
manufacturer column, so half the rows have it empty. Same field
configuration as Amazon-Google.
