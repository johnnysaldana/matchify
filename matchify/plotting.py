import os


def save_pr_curve(sweep_results, output_path, title=None):
    """
    Plot precision & recall curves for one or more models on a single
    figure and write to output_path.

    sweep_results is a list of (label, sweep_df) tuples
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, sweep_df in sweep_results:
        # plot recall on x, precision on y. sort by recall
        df = sweep_df.sort_values('recall')
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
