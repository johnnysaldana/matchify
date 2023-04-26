#!/usr/bin/env python

import os
import warnings

import click
from jinja2 import Environment, FileSystemLoader

from matchify.datasets import DATASETS


@click.group()
def cli():
    pass


def generate_html_table(dataset_results):
    file_loader = FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
    env = Environment(loader=file_loader)
    template = env.get_template("table.html")
    return template.render(dataset_results=dataset_results)


def _build_model(model_name, df, ignored_columns, field_config, blocking_config):
    # Each model gets its own copy: ExactMatchModel mutates self.df during
    # clustering and FlexMatchModel's preprocess() rewrites field columns.
    df = df.copy()
    if model_name == "exact":
        from matchify.models.exact_match_model import ExactMatchModel
        return ExactMatchModel(df, ignored_columns=ignored_columns)
    if model_name == "flex":
        from matchify.models.flex_match_model import FlexMatchModel
        return FlexMatchModel(
            df,
            field_config=field_config,
            blocking_config=blocking_config,
            ignored_columns=ignored_columns,
        )
    if model_name == "mlp":
        from matchify.models.mlp_match_model import MLPMatchModel
        return MLPMatchModel(
            df,
            field_config=field_config,
            blocking_config=blocking_config,
            ignored_columns=ignored_columns,
        )
    if model_name == "bert":
        from matchify.models.bert_match_model import BertMatchModel
        return BertMatchModel(
            df,
            field_config=field_config,
            blocking_config=blocking_config,
            ignored_columns=ignored_columns,
        )
    raise click.BadParameter(
        f"Unknown model '{model_name}'. Pick from: exact, flex, mlp, bert."
    )


@cli.command("model-comparisons")
@click.option(
    "--dataset", "datasets", multiple=True, default=("amazon-google",),
    help="Bundled dataset key (repeatable). One of: " + ", ".join(DATASETS),
)
@click.option(
    "--models", "models", multiple=True, default=("exact", "flex"),
    help="Models to evaluate (repeatable). One of: exact, flex, mlp, bert. "
         "bert requires the [deep] extra.",
)
@click.option(
    "--limit", default=None, type=int,
    help="Truncate each dataset to this many rows (handy for quick runs).",
)
@click.option(
    "--output", "output_path", default="output.html",
    help="HTML report path.",
)
@click.option(
    "--threshold", default=0.5, type=float,
    help="Score threshold used for the confusion matrix.",
)
@click.option(
    "--confusion/--no-confusion", default=True,
    help="Compute and emit the confusion matrix (slower; uses --threshold).",
)
def model_comparisons(datasets, models, limit, output_path, threshold, confusion):
    """Run the configured models on the configured datasets and write an HTML report."""
    import pandas as pd

    warnings.filterwarnings("ignore")

    dataset_results = []
    for dataset_key in datasets:
        if dataset_key not in DATASETS:
            raise click.BadParameter(
                f"Unknown dataset '{dataset_key}'. Pick from: {', '.join(DATASETS)}"
            )
        cfg = DATASETS[dataset_key]
        click.echo(f"\n=== {cfg['label']} ===")
        df = pd.read_csv(cfg["path"])
        if limit:
            df = df.head(limit)
        ignored_columns = cfg["ignored_columns"]

        # Pick a sample lookup record from the dataset's configured lookup_id
        record = df[df["id"] == cfg["lookup_id"]].iloc[0]
        record = record[[x for x in record.index if x not in ignored_columns]]

        model_results = []
        for model_name in models:
            click.echo(f"  - {model_name}: training/predicting...")
            model = _build_model(
                model_name, df, ignored_columns, cfg["field_config"], cfg["blocking_config"]
            )
            if hasattr(model, "train"):
                model.train()
            predictions = model.predict(
                record, only_matches=False, return_full_record=True
            ).head(10)
            mrr = model.mrr()
            click.echo(f"    MRR: {mrr:.4f}")
            confusion_stats = None
            if confusion:
                confusion_stats = model.confusion_matrix(threshold=threshold)
                click.echo(
                    f"    confusion@{threshold}: "
                    f"tp={confusion_stats['tp']} fp={confusion_stats['fp']} "
                    f"tn={confusion_stats['tn']} fn={confusion_stats['fn']} "
                    f"P={confusion_stats['precision']:.3f} "
                    f"R={confusion_stats['recall']:.3f} "
                    f"F1={confusion_stats['f1']:.3f}"
                )
            model_results.append({
                "model_name": type(model).__name__,
                "predictions": predictions,
                "mrr_score": mrr,
                "confusion": confusion_stats,
            })

        dataset_results.append({
            "dataset_name": cfg["label"],
            "sample_record": record,
            "model_results": model_results,
        })

    html_output = generate_html_table(dataset_results)
    with open(output_path, "w") as f:
        f.write(html_output)

    click.echo(f"\nGenerated {output_path}")


if __name__ == "__main__":
    cli()
