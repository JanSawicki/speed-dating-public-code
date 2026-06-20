import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agent import Agent, Match
from analysis._common import TIERS, _iter_matches, _model_slug, _subject_columns


def build_variance_df(attendees: List[Agent], subjects: List[Agent], matches: Dict[str, List[Match]]) -> pd.DataFrame:
    columns = _subject_columns(subjects)
    values: Dict[str, Dict[str, list]] = {tier: {col: [] for col in columns} for tier in TIERS}
    for _, tier, col, match in _iter_matches(attendees, subjects, matches):
        values[tier][col].append(float(match.compatibility_score >= 50))
    std_df = pd.DataFrame(float("nan"), index=TIERS, columns=columns)
    for tier in TIERS:
        for col in columns:
            v = values[tier][col]
            if len(v) > 1:
                std_df.loc[tier, col] = pd.Series(v).std()
            elif len(v) == 1:
                std_df.loc[tier, col] = 0.0
    return std_df


def save_variance_heatmap(variance_df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "variance")
    os.makedirs(subdir, exist_ok=True)
    df = variance_df.astype(float)

    fig, ax = plt.subplots(figsize=(len(df.columns) * 2 + 2, len(df.index) * 1.5 + 2))
    sns.heatmap(
        df, ax=ax, annot=True, fmt=".2f",
        cmap="Purples", vmin=0.0, vmax=0.5,
        linewidths=0.5,
        cbar_kws={"label": "Std dev of is_compatible across runs"},
    )
    ax.set_title(f"Compatibility rate variance\n{model_id}", pad=12)
    ax.set_xlabel("Subject prompt level")
    ax.set_ylabel("Subject compatibility tier")
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, f"{_model_slug(model_id)}_variance.pdf"), format="pdf")
    plt.close(fig)


def save_variance_heatmap_all_models(
    variance_dfs: Dict[str, pd.DataFrame], charts_path: str
):
    subdir = os.path.join(charts_path, "variance")
    os.makedirs(subdir, exist_ok=True)
    models = list(variance_dfs.keys())
    n = len(models)
    any_df = next(iter(variance_dfs.values()))
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
        df = variance_dfs[model_id].astype(float)
        sns.heatmap(
            df, ax=ax, annot=True, fmt=".2f",
            cmap="Purples", vmin=0.0, vmax=0.5,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(model_id, pad=12)
        ax.set_xlabel("Subject prompt level")
        ax.set_ylabel("Subject compatibility tier" if col == 0 else "")

    sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0.0, vmax=0.5))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Std dev of is_compatible across runs")

    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.suptitle("Compatibility rate variance — all models", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "variance_all_models.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_variance_md(variance_df: pd.DataFrame, ranking_df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "variance")
    os.makedirs(subdir, exist_ok=True)
    lines = [
        "# Compatibility rate variance",
        f"## {model_id}",
        "",
        "### Mean",
        "",
        ranking_df.astype(float).to_markdown(floatfmt=".2f"),
        "",
        "### Standard deviation",
        "",
        variance_df.astype(float).to_markdown(floatfmt=".2f"),
    ]
    with open(os.path.join(subdir, f"{_model_slug(model_id)}_variance.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_variance_md_all_models(
    variance_dfs: Dict[str, pd.DataFrame],
    ranking_dfs: Dict[str, pd.DataFrame],
    charts_path: str,
):
    subdir = os.path.join(charts_path, "variance")
    os.makedirs(subdir, exist_ok=True)
    lines = ["# Compatibility rate variance — all models", ""]
    for model_id in variance_dfs:
        lines += [
            f"## {model_id}",
            "",
            "### Mean",
            "",
            ranking_dfs[model_id].astype(float).to_markdown(floatfmt=".2f"),
            "",
            "### Standard deviation",
            "",
            variance_dfs[model_id].astype(float).to_markdown(floatfmt=".2f"),
            "",
        ]
    with open(os.path.join(subdir, "variance_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
