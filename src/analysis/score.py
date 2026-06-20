import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agent import Agent, Match
from analysis._common import TIERS, _iter_matches, _model_slug, _subject_columns


def build_score_ranking_df(
    attendees: List[Agent], subjects: List[Agent], matches: Dict[str, List[Match]]
) -> pd.DataFrame:
    columns = _subject_columns(subjects)
    score_sum = pd.DataFrame(0.0, index=TIERS, columns=columns)
    counts = pd.DataFrame(0.0, index=TIERS, columns=columns)
    for _, tier, col, match in _iter_matches(attendees, subjects, matches):
        score_sum.loc[tier, col] += float(match.compatibility_score)
        counts.loc[tier, col] += 1
    return score_sum / counts.replace(0.0, float("nan"))


def save_score_heatmap(score_df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    df = score_df.astype(float)

    fig, ax = plt.subplots(figsize=(len(df.columns) * 2 + 2, len(df.index) * 1.5 + 2))
    sns.heatmap(
        df, ax=ax, annot=True, fmt=".1f",
        cmap="RdYlGn", vmin=0.0, vmax=100.0,
        linewidths=0.5,
        cbar_kws={"label": "Mean compatibility score (0–100)"},
    )
    ax.set_title(f"Mean compatibility score\n{model_id}", pad=12)
    ax.set_xlabel("Subject prompt level")
    ax.set_ylabel("Subject compatibility tier")
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, f"{_model_slug(model_id)}_score_heatmap.pdf"), format="pdf")
    plt.close(fig)


def save_score_heatmap_all_models(score_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    models = list(score_dfs.keys())
    n = len(models)
    any_df = next(iter(score_dfs.values()))
    cell_w, cell_h = 2, 1.5

    fig, all_axes = plt.subplots(
        1, n + 1,
        figsize=(n * 3.36, 3.84),
        gridspec_kw={'width_ratios': [1] * n + [0.08]},
    )
    axes = list(all_axes[:n])
    cbar_ax = all_axes[-1]
    for ax in axes[1:]:
        ax.sharey(axes[0])

    for col, (model_id, ax) in enumerate(zip(models, axes)):
        df = score_dfs[model_id].astype(float)
        sns.heatmap(
            df, ax=ax, annot=True, fmt=".1f",
            cmap="RdYlGn", vmin=0.0, vmax=100.0,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(model_id, pad=12)
        ax.set_xlabel("Subject prompt level")
        ax.set_ylabel("Subject compatibility tier" if col == 0 else "")

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0.0, vmax=100.0))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Mean compatibility score (0–100)")

    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.suptitle("Mean compatibility score — all models", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "score_heatmap_all_models.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_score_md_all_models(score_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    lines = ["# Mean compatibility score — all models", ""]
    for model_id, score_df in score_dfs.items():
        lines += [f"## {model_id}", "", score_df.astype(float).to_markdown(floatfmt=".1f"), ""]
    with open(os.path.join(subdir, "score_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_score_md(score_df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    lines = [
        "# Mean compatibility score",
        f"## {model_id}",
        "",
        score_df.astype(float).to_markdown(floatfmt=".1f"),
    ]
    with open(os.path.join(subdir, f"{_model_slug(model_id)}_score.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
