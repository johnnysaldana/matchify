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
    # each model gets its own copy. ExactMatchModel mutates self.df,
    # FlexMatchModel rewrites columns.
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
    if model_name == "siamese":
        from matchify.models.siamese_match_model import SiameseMatchModel
        return SiameseMatchModel(
            df,
            field_config=field_config,
            blocking_config=blocking_config,
            ignored_columns=ignored_columns,
        )
    raise click.BadParameter(
        f"Unknown model '{model_name}'. Pick from: exact, flex, mlp, bert, siamese."
    )


ALL_MODELS = ("exact", "flex", "mlp", "bert", "siamese")


def _available_models(requested):
    """Drop bert/siamese if sentence_transformers isn't installed."""
    import importlib.util
    has_deep = importlib.util.find_spec("sentence_transformers") is not None
    out = []
    for m in requested:
        if m in ("bert", "siamese") and not has_deep:
            click.echo(
                f"  (skipping '{m}': requires the [deep] extra. "
                f"Install with `pip install matchify[deep]`.)",
                err=True,
            )
            continue
        out.append(m)
    return tuple(out)


@cli.command("model-comparisons")
@click.option(
    "--dataset", "datasets", multiple=True, default=("amazon-google",),
    help="Bundled dataset key (repeatable). One of: " + ", ".join(DATASETS),
)
@click.option(
    "--models", "models", multiple=True, default=("exact", "flex"),
    help="Models to evaluate (repeatable). One of: " + ", ".join(ALL_MODELS) + ". "
         "bert and siamese require the [deep] extra.",
)
@click.option(
    "--all", "run_all", is_flag=True, default=False,
    help="Run every bundled dataset against every available model. "
         "Overrides --dataset and --models. The deep models are skipped "
         "automatically if the [deep] extra isn't installed.",
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
@click.option(
    "--pr-curves", "pr_curves_dir", default=None, type=str,
    help="Directory to write per-dataset precision/recall PNGs to. "
         "Sweeps thresholds 0..1 in 0.02 steps for every model.",
)
def model_comparisons(datasets, models, run_all, limit, output_path, threshold, confusion, pr_curves_dir):
    """Run the configured models on the configured datasets, write HTML report.

    Try everything on a 500-row slice:

        matchify model-comparisons --all --limit 500
    """
    import pandas as pd

    warnings.filterwarnings("ignore")

    if run_all:
        datasets = tuple(DATASETS.keys())
        models = ALL_MODELS

    models = _available_models(models)
    if not models:
        raise click.ClickException("No runnable models. Install [deep] or pick from: exact, flex, mlp.")

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

        # sample lookup record for the predictions table
        record = df[df["id"] == cfg["lookup_id"]].iloc[0]
        record = record[[x for x in record.index if x not in ignored_columns]]

        model_results = []
        sweep_results = []
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
            if pr_curves_dir:
                # one walk produces both the per-threshold sweep and the
                # confusion matrix at the requested threshold. cheaper
                # than running them separately.
                sweep_df = model.threshold_sweep()
                sweep_results.append((type(model).__name__, sweep_df))
                if confusion:
                    closest = (sweep_df['threshold'] - threshold).abs().idxmin()
                    confusion_stats = sweep_df.loc[closest].to_dict()
            elif confusion:
                confusion_stats = model.confusion_matrix(threshold=threshold)
            if confusion_stats:
                click.echo(
                    f"    confusion@{threshold}: "
                    f"tp={int(confusion_stats['tp'])} fp={int(confusion_stats['fp'])} "
                    f"tn={int(confusion_stats['tn'])} fn={int(confusion_stats['fn'])} "
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

        if pr_curves_dir and sweep_results:
            from matchify.plotting import save_pr_curve
            png_path = os.path.join(pr_curves_dir, f"pr_{dataset_key}.png")
            save_pr_curve(sweep_results, png_path, title=cfg['label'])
            click.echo(f"    PR curve: {png_path}")

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
