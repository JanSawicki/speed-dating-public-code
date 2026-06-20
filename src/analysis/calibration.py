import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._common import TIERS, _model_slug


def save_calibration_curve(ranking_df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "calibration")
    os.makedirs(subdir, exist_ok=True)

    actual_overlap = {"COMPATIBLE": 1.0, "PARTIAL": 0.5, "INCOMPATIBLE": 0.0}
    tier_colors = {"COMPATIBLE": "#2ecc71", "PARTIAL": "#f39c12", "INCOMPATIBLE": "#e74c3c"}
    df = ranking_df.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="Perfect calibration")

    for col in df.columns:
        xs = [actual_overlap[tier] for tier in TIERS]
        ys = [df.loc[tier, col] for tier in TIERS]
        ax.plot(xs, ys, marker="o", label=col)
        for tier, x, y in zip(TIERS, xs, ys):
            if not np.isnan(y):
                ax.annotate(tier[:4], (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7,
                            color=tier_colors[tier])

    ax.set_xlabel("Actual hobby overlap")
    ax.set_ylabel("Compatibility rate")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Pressure level")
    ax.set_title(f"Calibration curve\n{model_id}", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, f"{_model_slug(model_id)}_calibration.pdf"), format="pdf")
    plt.close(fig)


def save_calibration_curve_all_models(ranking_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "calibration")
    os.makedirs(subdir, exist_ok=True)
    models = list(ranking_dfs.keys())
    n = len(models)

    actual_overlap = {"COMPATIBLE": 1.0, "PARTIAL": 0.5, "INCOMPATIBLE": 0.0}
    tier_colors = {"COMPATIBLE": "#2ecc71", "PARTIAL": "#f39c12", "INCOMPATIBLE": "#e74c3c"}

    fig, axes = plt.subplots(1, n, figsize=(n * 3.36, 3.84), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for col, (model_id, ax) in enumerate(zip(models, axes)):
        df = ranking_dfs[model_id].astype(float)
        ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="Perfect calibration")
        for level_col in df.columns:
            xs = [actual_overlap[tier] for tier in TIERS]
            ys = [df.loc[tier, level_col] for tier in TIERS]
            ax.plot(xs, ys, marker="o", label=level_col)
            for tier, x, y in zip(TIERS, xs, ys):
                if not np.isnan(y):
                    ax.annotate(tier[:4], (x, y), textcoords="offset points", xytext=(5, 3),
                                fontsize=7, color=tier_colors[tier])
        ax.set_xlabel("Actual hobby overlap")
        ax.set_ylabel("Compatibility rate" if col == 0 else "")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Pressure level")
        ax.set_title(model_id, pad=12)

    fig.suptitle("Calibration curve — all models", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "calibration_all_models.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_calibration_md_all_models(ranking_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "calibration")
    os.makedirs(subdir, exist_ok=True)

    actual_overlap = {"COMPATIBLE": 1.0, "PARTIAL": 0.5, "INCOMPATIBLE": 0.0}
    lines = ["# Calibration curve — all models", ""]
    for model_id, ranking_df in ranking_dfs.items():
        df = ranking_df.astype(float).copy()
        df.insert(0, "Actual overlap", [actual_overlap[t] for t in TIERS])
        lines += [f"## {model_id}", "", df.to_markdown(floatfmt=".2f"), ""]
    with open(os.path.join(subdir, "calibration_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_calibration_md(ranking_df: pd.DataFrame, charts_path: str, model_id: str):
    subdir = os.path.join(charts_path, "calibration")
    os.makedirs(subdir, exist_ok=True)

    actual_overlap = {"COMPATIBLE": 1.0, "PARTIAL": 0.5, "INCOMPATIBLE": 0.0}
    df = ranking_df.astype(float)
    cal_df = df.copy()
    cal_df.insert(0, "Actual overlap", [actual_overlap[t] for t in TIERS])

    lines = [
        "# Calibration curve",
        f"## {model_id}",
        "",
        cal_df.to_markdown(floatfmt=".2f"),
    ]
    with open(os.path.join(subdir, f"{_model_slug(model_id)}_calibration.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
