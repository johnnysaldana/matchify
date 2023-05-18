"""
Registry of bundled benchmark datasets.

Each entry pairs a Combined*.csv (one row per record, with id/group_id
columns) with default field/blocking configs for that dataset. The CLI
uses these to drive multi-dataset comparison runs.
"""

import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


DATASETS = {
    "amazon-google": {
        "label": "Amazon-Google Products",
        "path": os.path.join(REPO_ROOT, "datasets/Amazon-GoogleProducts/CombinedProducts.csv"),
        "ignored_columns": ["id", "group_id", "original_id"],
        "lookup_id": 8,
        "field_config": {
            "name":         {"type": "other", "comparison_method": "jaro_winkler"},
            "description":  {"type": "other", "comparison_method": "tfidf_cosine"},
            "manufacturer": {"type": "other", "comparison_method": "jaro_winkler"},
            "price":        {"type": "other", "comparison_method": "jaro_winkler"},
        },
        "blocking_config": {
            "name": {"method": "prefix", "threshold": 2},
        },
    },
    "dblp-acm": {
        "label": "DBLP-ACM",
        "path": os.path.join(REPO_ROOT, "datasets/DBLP-ACM/CombinedAcademic.csv"),
        "ignored_columns": ["id", "group_id", "original_id"],
        "lookup_id": 2,
        "field_config": {
            "title":   {"type": "other", "comparison_method": "tfidf_cosine"},
            "authors": {"type": "other", "comparison_method": "jaro_winkler"},
            "venue":   {"type": "other", "comparison_method": "jaro_winkler"},
            "year":    {"type": "other", "comparison_method": "jaro_winkler"},
        },
        "blocking_config": {
            "title": {"method": "prefix", "threshold": 3},
        },
    },
    "synthetic-people": {
        "label": "Synthetic People",
        "path": os.path.join(REPO_ROOT, "datasets/SyntheticPeople/CombinedPeople.csv"),
        "ignored_columns": ["id", "group_id"],
        "lookup_id": 1,
        # only bundled dataset that uses type='name'/'phone'/'address'/'date'.
        # the real-world benchmarks above use 'other' and skip the normalizers.
        "field_config": {
            "first_name":   {"type": "name",    "comparison_method": "jaro_winkler"},
            "last_name":    {"type": "name",    "comparison_method": "jaro_winkler"},
            "birthdate":    {"type": "date",    "comparison_method": "jaro_winkler"},
            "address":      {"type": "address", "comparison_method": "tfidf_cosine"},
            "phone_number": {"type": "phone",   "comparison_method": "jaro_winkler"},
        },
        "blocking_config": {
            "last_name": {"method": "prefix", "threshold": 2},
        },
    },
}
