import os


def save_pr_curve(sweep_results, output_path, title=None):
    """
    Plot precision & recall curves for one or more models on a single
    figure and write to output_path.

    sweep_results is a list of (label, sweep_df) tuples. Models whose
    sweep collapses to a single non-trivial operating point (e.g.,
    ExactMatchModel produces only 0/1 scores so there is one decision
    point and two boundary points) are rendered as a marker rather than
    a line. Connecting those points with a line implies a precision-
    recall trade-off the model cannot actually make.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _operating_points(df):
        # rows that are not the (recall=0, precision=0) sentinel and
        # not the (recall=1, precision=base_rate) recall-everything row.
        # what's left are the model's real decision points.
        if df.empty:
            return df
        return df[(df['recall'] > 0.0) & (df['recall'] < 1.0)]

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, sweep_df in sweep_results:
        df = sweep_df.sort_values('recall').reset_index(drop=True)
        ops = _operating_points(df)
        if len(ops) <= 1:
            # binary scorer: plot the single operating point as a marker.
            point = ops if not ops.empty else df.iloc[[df['f1'].idxmax()]]
            ax.scatter(
                point['recall'], point['precision'],
                marker='o', s=70, label=label, zorder=3,
            )
            continue
        ax.plot(df['recall'], df['precision'], marker='.', label=label, linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
