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


def _build_model(
    model_name, df, ignored_columns, field_config, blocking_config,
    test_size=0.0, random_state=0,
):
    # each model gets its own copy. ExactMatchModel mutates self.df,
    # FlexMatchModel rewrites columns.
    df = df.copy()
    common = {
        "ignored_columns": ignored_columns,
        "test_size": test_size,
        "random_state": random_state,
    }
    if model_name == "exact":
        from matchify.models.exact_match_model import ExactMatchModel
        return ExactMatchModel(df, **common)
    if model_name == "flex":
        from matchify.models.flex_match_model import FlexMatchModel
        return FlexMatchModel(df, field_config=field_config, blocking_config=blocking_config, **common)
    if model_name == "mlp":
        from matchify.models.mlp_match_model import MLPMatchModel
        return MLPMatchModel(df, field_config=field_config, blocking_config=blocking_config, **common)
    if model_name == "bert":
        from matchify.models.bert_match_model import BertMatchModel
        return BertMatchModel(df, field_config=field_config, blocking_config=blocking_config, **common)
    if model_name == "siamese":
        from matchify.models.siamese_match_model import SiameseMatchModel
        return SiameseMatchModel(df, field_config=field_config, blocking_config=blocking_config, **common)
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
         "Overrides --dataset and --models. Deep models are skipped if "
         "[deep] isn't installed.",
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
    help="Compute and emit the confusion matrix (slower, uses --threshold).",
)
@click.option(
    "--pr-curves", "pr_curves_dir", default=None, type=str,
    help="Directory to write per-dataset precision/recall PNGs to. "
         "Sweeps thresholds 0..1 in 0.02 steps for every model.",
)
@click.option(
    "--test-size", default=0.0, type=float,
    help="Fraction of groups to hold out for eval. Supervised models "
         "only train on the train partition. 0 means no split.",
)
@click.option(
    "--random-state", default=0, type=int,
    help="Seed for the split + model RNGs. With --seeds N, uses "
         "random_state, +1, ..., +N-1.",
)
@click.option(
    "--seeds", default=1, type=int,
    help="Number of seeds for stochastic models (MLP, Siamese). Each "
         "seed gets its own train run. MRR and F1 reported as mean +/- std. "
         "Deterministic models (Exact, Flex, BERT) ignore this flag.",
)
def model_comparisons(
    datasets, models, run_all, limit, output_path, threshold, confusion,
    pr_curves_dir, test_size, random_state, seeds,
):
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

        from matchify.evaluation import aggregate_metric, aggregate_sweeps, is_stochastic

        model_results = []
        sweep_results = []
        for model_name in models:
            n_runs = seeds if is_stochastic(model_name) else 1
            click.echo(f"  - {model_name}: {n_runs} seed{'s' if n_runs > 1 else ''}")

            mrr_runs = []
            sweep_runs = []
            confusion_runs = []
            predictions = None
            class_label = None
            for seed_idx in range(n_runs):
                seed = random_state + seed_idx
                model = _build_model(
                    model_name, df, ignored_columns, cfg["field_config"], cfg["blocking_config"],
                    test_size=test_size, random_state=seed,
                )
                if hasattr(model, "train"):
                    model.train()
                if predictions is None:
                    predictions = model.predict(
                        record, only_matches=False, return_full_record=True
                    ).head(10)
                    class_label = type(model).__name__
                mrr_runs.append(model.mrr())
                if pr_curves_dir or confusion:
                    # threshold_sweep caches the scored pairs internally,
                    # so confusion_matrix at the user-specified threshold
                    # reuses them without re-running predict.
                    sweep_runs.append(model.threshold_sweep())
                    if confusion:
                        confusion_runs.append(model.confusion_matrix(threshold))

            mrr_mean, mrr_std = aggregate_metric(mrr_runs)
            click.echo(f"    MRR: {mrr_mean:.4f} ± {mrr_std:.4f}" if n_runs > 1 else f"    MRR: {mrr_mean:.4f}")

            confusion_stats = None
            sweep_df = None
            if sweep_runs:
                sweep_df = aggregate_sweeps(sweep_runs)
                if confusion_runs:
                    # average the per-seed confusion stats at the
                    # user-specified threshold rather than reading off
                    # the aggregated PR curve (whose threshold axis is
                    # not aligned across seeds).
                    p_mean, _ = aggregate_metric([c['precision'] for c in confusion_runs])
                    r_mean, _ = aggregate_metric([c['recall'] for c in confusion_runs])
                    f1_mean, f1_std = aggregate_metric([c['f1'] for c in confusion_runs])
                    confusion_stats = {
                        'threshold': threshold,
                        'precision': p_mean,
                        'recall': r_mean,
                        'f1': f1_mean,
                        'f1_std': f1_std,
                    }
                if pr_curves_dir:
                    sweep_results.append((class_label, sweep_df))

            if confusion_stats:
                pm_std = f" ± {confusion_stats['f1_std']:.3f}" if confusion_stats.get('f1_std') else ""
                click.echo(
                    f"    confusion@{threshold}: "
                    f"P={confusion_stats['precision']:.3f} "
                    f"R={confusion_stats['recall']:.3f} "
                    f"F1={confusion_stats['f1']:.3f}{pm_std}"
                )
            model_results.append({
                "model_name": class_label,
                "predictions": predictions,
                "mrr_score": mrr_mean,
                "mrr_std": mrr_std,
                "n_seeds": n_runs,
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
