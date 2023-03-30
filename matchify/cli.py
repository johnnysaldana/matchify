#!/usr/bin/env python

import click
from jinja2 import Environment, FileSystemLoader
import os


@click.group()
def cli():
    pass


def generate_html_table(dataset_name, sample_record, model_results):
    file_loader = FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
    env = Environment(loader=file_loader)
    template = env.get_template("table.html")

    output = template.render(
        dataset_name=dataset_name,
        sample_record=sample_record,
        model_results=model_results,
    )

    return output


@cli.command()
def model_comparisons():
    from matchify.models.exact_match_model import ExactMatchModel
    from matchify.models.flex_match_model import FlexMatchModel
    import numpy as np
    import pandas as pd

    df = pd.read_csv('datasets/Amazon-GoogleProducts/CombinedProducts.csv')[:100]  # [:100] used to run this quickly
    ignored_columns = ['id', 'group_id', 'original_id']
    record_to_predict = df[df['id'] == 8].iloc[0]  # b00002s5ig (8)
    record_to_predict = record_to_predict[[x for x in record_to_predict.index if x not in ignored_columns]]

    EMM = ExactMatchModel(df, ignored_columns=ignored_columns)
    predictions_EMM = EMM.predict(record_to_predict, only_matches=False, return_full_record=True).head(10)
    mrr_EMM = EMM.mrr()

    field_config = {
        "name": {"type": "other", "comparison_method": "jaro_winkler"},
        "description": {"type": "other", "comparison_method": "tfidf_cosine"},
        "manufacturer": {"type": "other", "comparison_method": "jaro_winkler"},
        "price": {"type": "other", "comparison_method": "jaro_winkler"},
    }

    blocking_config = {
        "name": {"method": "prefix", "threshold": 2}
    }
    FMM = FlexMatchModel(df, field_config=field_config, blocking_config=blocking_config, ignored_columns=ignored_columns)
    FMM.train()
    predictions_FMM = FMM.predict(record_to_predict, only_matches=False, return_full_record=True).head(10)
    mrr_FMM = FMM.mrr()
    model_results = [
        {
            "model_name": "ExactMatchModel",
            "predictions": predictions_EMM,
            "mrr_score": mrr_EMM,
        },
        {
            "model_name": "FlexMatchModel",
            "predictions": predictions_FMM,
            "mrr_score": mrr_FMM,
        },
    ]

    dataset_name = "Amazon-GoogleProducts"
    sample_record = record_to_predict

    html_output = generate_html_table(dataset_name, sample_record, model_results)

    with open("output.html", "w") as f:
        f.write(html_output)

    print("Generated output.html")


if __name__ == "__main__":
    cli()
