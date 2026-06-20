import os
from itertools import combinations
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_cross_model_heatmap(ranking_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "cross_model")
    os.makedirs(subdir, exist_ok=True)
    models = list(ranking_dfs.keys())
    pairs = list(combinations(models, 2))
    n = len(pairs)

    diffs = {
        (model_a, model_b): (
            ranking_dfs[model_a].astype(float) - ranking_dfs[model_b].astype(float)
        ).loc[["INCOMPATIBLE"]]
        for model_a, model_b in pairs
    }
    diff_abs_max = max(diff.abs().max().max() for diff in diffs.values())
    diff_abs_max = max(diff_abs_max, 0.1)

    fig, all_axes = plt.subplots(1, n + 1, figsize=(n * 3.9, 2.6),
                                  gridspec_kw={'width_ratios': [1] * n + [0.08]})
    axes = list(all_axes[:n])
    cbar_ax = all_axes[-1]
    for ax in axes[1:]:
        ax.sharey(axes[0])

    for col, ((model_a, model_b), ax) in enumerate(zip(pairs, axes)):
        sns.heatmap(
            diffs[(model_a, model_b)], ax=ax, annot=True, fmt="+.2f",
            cmap="RdBu_r", vmin=-diff_abs_max, vmax=diff_abs_max,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(f"{model_a}\n− {model_b}", fontsize=8, linespacing=1.4)
        ax.set_xlabel("Subject prompt level")
        if col == 0:
            ax.set_ylabel("Incompatible tier")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-diff_abs_max, vmax=diff_abs_max))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Signed diff. (A − B)")

    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.suptitle(
        "Cross-model comparison — pairwise signed difference in Incompatible-tier compatibility rate",
        y=1.15, fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "cross_model_comparison.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)



def save_cross_model_md(ranking_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "cross_model")
    os.makedirs(subdir, exist_ok=True)
    models = list(ranking_dfs.keys())
    lines = ["# Cross-model comparison", ""]

    for model_id in models:
        df = ranking_dfs[model_id].astype(float)
        lines += [f"## {model_id}", "", df.to_markdown(floatfmt=".2f"), ""]

    lines += ["## Pairwise differences", ""]
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            diff = ranking_dfs[model_a].astype(float) - ranking_dfs[model_b].astype(float)
            lines += [f"### {model_a} − {model_b}", "", diff.to_markdown(floatfmt="+.2f"), ""]

    with open(os.path.join(subdir, "cross_model_comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
