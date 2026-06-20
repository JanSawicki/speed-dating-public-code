import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from agent import Agent, Match
from analysis._common import TIERS, _iter_matches, _model_slug, _subject_columns

_NEUTRAL_COLOR = "#4C72B0"


def build_score_distribution_df(
    attendees: List[Agent], subjects: List[Agent], matches: Dict[str, List[Match]]
) -> pd.DataFrame:
    rows = []
    for _, tier, col, match in _iter_matches(attendees, subjects, matches):
        rows.append({"tier": tier, "level": col, "score": float(match.compatibility_score)})
    return pd.DataFrame(rows, columns=["tier", "level", "score"])


def save_score_distribution_plot(df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    levels = _subject_columns_from_df(df)

    fig, axes = plt.subplots(
        len(TIERS), len(levels),
        figsize=(len(levels) * 2.2, len(TIERS) * 1.8),
        sharex=True, sharey=True,
    )
    for row, tier in enumerate(TIERS):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            cell_scores = df[(df["tier"] == tier) & (df["level"] == level)]["score"]
            ax.hist(cell_scores, bins=10, range=(0, 100), color=_NEUTRAL_COLOR, edgecolor="white")
            ax.axvline(50, color="black", linestyle="--", linewidth=0.8)
            if row == 0:
                ax.set_title(level)
            if col == 0:
                ax.set_ylabel(tier.title())
            if row == len(TIERS) - 1:
                ax.set_xlabel("Score")
            ax.set_xlim(0, 100)

    fig.suptitle(f"Raw compatibility score distribution\n{model_id}", y=1.02)
    plt.tight_layout()
    fig.savefig(
        os.path.join(subdir, f"{_model_slug(model_id)}_score_distribution.pdf"),
        format="pdf", bbox_inches="tight",
    )
    plt.close(fig)


def save_score_distribution_plot_all_models(dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    models = list(dfs.keys())
    levels = _subject_columns_from_df(next(iter(dfs.values())))
    n_models, n_levels, n_tiers = len(models), len(levels), len(TIERS)

    fig, axes = plt.subplots(
        n_tiers, n_models * n_levels,
        figsize=(n_models * n_levels * 1.3, n_tiers * 1.6),
        sharex=True, sharey="row",
    )
    for m, model_id in enumerate(models):
        df = dfs[model_id]
        for row, tier in enumerate(TIERS):
            for col, level in enumerate(levels):
                ax = axes[row, m * n_levels + col]
                cell_scores = df[(df["tier"] == tier) & (df["level"] == level)]["score"]
                ax.hist(cell_scores, bins=10, range=(0, 100), color=_NEUTRAL_COLOR, edgecolor="white")
                ax.axvline(50, color="black", linestyle="--", linewidth=0.8)
                ax.set_xlim(0, 100)
                if row == 0:
                    ax.set_title(level.replace("Level ", "L"), fontsize=8)
                if col == 0 and m == 0:
                    ax.set_ylabel(tier.title())
                if row == n_tiers - 1:
                    ax.set_xticks([0, 50, 100])
                    ax.tick_params(axis="x", labelsize=6)

    for m, model_id in enumerate(models):
        x0 = axes[0, m * n_levels].get_position().x0
        x1 = axes[0, m * n_levels + n_levels - 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.98, model_id, ha="center", va="bottom", fontsize=9)

    fig.suptitle("Raw compatibility score distribution — all models", y=1.06)
    fig.savefig(
        os.path.join(subdir, "score_distribution_all_models.pdf"),
        format="pdf", bbox_inches="tight",
    )
    plt.close(fig)


def _subject_columns_from_df(df: pd.DataFrame) -> List[str]:
    return sorted(df["level"].unique(), key=lambda l: int(l.split(" ")[1]))


def _mass_at_50(df: pd.DataFrame) -> pd.DataFrame:
    levels = _subject_columns_from_df(df)
    mass = pd.DataFrame(float("nan"), index=TIERS, columns=levels)
    for tier in TIERS:
        for level in levels:
            cell = df[(df["tier"] == tier) & (df["level"] == level)]["score"]
            if len(cell) > 0:
                mass.loc[tier, level] = float((cell == 50).mean())
    return mass


def save_mass_at_50_md(df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    mass = _mass_at_50(df)
    lines = [
        "# Fraction of observations with score exactly 50",
        f"## {model_id}",
        "",
        mass.to_markdown(floatfmt=".2f"),
    ]
    out_path = os.path.join(subdir, f"{_model_slug(model_id)}_mass_at_50.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_mass_at_50_md_all_models(dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "compatibility_score")
    os.makedirs(subdir, exist_ok=True)
    lines = ["# Fraction of observations with score exactly 50 — all models", ""]
    for model_id, df in dfs.items():
        mass = _mass_at_50(df)
        lines += [f"## {model_id}", "", mass.to_markdown(floatfmt=".2f"), ""]
    with open(os.path.join(subdir, "mass_at_50_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
