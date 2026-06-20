import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from analysis._common import TIERS, _model_slug


def save_pressure_response_chart_all_models(ranking_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "pressure_response")
    os.makedirs(subdir, exist_ok=True)

    n = len(ranking_dfs)
    fig, ax = plt.subplots(figsize=(n * 3.36, 3.84))
    for model_id, ranking_df in ranking_dfs.items():
        df = ranking_df.astype(float)
        levels = [c for c in df.columns]
        xs = [int(c.split()[-1]) for c in levels]
        ys = [df.loc["INCOMPATIBLE", c] for c in levels]
        ax.plot(xs, ys, marker="o", label=model_id)

    ax.set_xlabel("Pressure level")
    ax.set_ylabel("Compatibility rate (INCOMPATIBLE tier)")
    ax.set_xticks([int(c.split()[-1]) for c in next(iter(ranking_dfs.values())).columns])
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Pressure-response curve — INCOMPATIBLE tier", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "pressure_response_all_models.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_pressure_response_md_all_models(ranking_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "pressure_response")
    os.makedirs(subdir, exist_ok=True)

    lines = ["# Pressure-response curve — INCOMPATIBLE tier", ""]
    for model_id, ranking_df in ranking_dfs.items():
        df = ranking_df.astype(float)
        row = df.loc[["INCOMPATIBLE"]]
        lines += [f"## {model_id}", "", row.to_markdown(floatfmt=".2f"), ""]

    with open(os.path.join(subdir, "pressure_response_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
