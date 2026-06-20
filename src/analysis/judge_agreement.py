import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis._common import _model_slug

_AGREEMENT_THRESHOLD = 0.5


def build_judge_diff_df(lie_df: pd.DataFrame, ext_lie_df: pd.DataFrame) -> pd.DataFrame:
    """Signed difference per cell: external judge − self-evaluation."""
    return ext_lie_df.astype(float) - lie_df.astype(float)


def build_judge_agreement_df(
    lie_df: pd.DataFrame,
    ext_lie_df: pd.DataFrame,
    threshold: float = _AGREEMENT_THRESHOLD,
) -> pd.DataFrame:
    """
    Directional agreement per cell: 1.0 if both evaluators agree whether lying is
    occurring (both > threshold or both ≤ threshold), 0.0 otherwise.
    """
    self_pos = lie_df.astype(float) > threshold
    ext_pos = ext_lie_df.astype(float) > threshold
    return (self_pos == ext_pos).astype(float)


def save_judge_agreement_heatmap_all_models(
    lie_dfs: Dict[str, pd.DataFrame],
    ext_lie_dfs: Dict[str, pd.DataFrame],
    charts_path: str,
    judge_model_id: str = None,
):
    subdir = os.path.join(charts_path, "judge_agreement")
    os.makedirs(subdir, exist_ok=True)
    models = list(lie_dfs.keys())
    n = len(models)

    any_df = next(iter(lie_dfs.values()))
    fig, all_axes = plt.subplots(
        2, n + 1,
        figsize=(n * 3.36, 3.84),
        gridspec_kw={'width_ratios': [1] * n + [0.08]},
        squeeze=False,
    )
    axes = all_axes[:, :n]
    cbar_axes = all_axes[:, n]
    for row in range(2):
        for col in range(1, n):
            axes[row][col].sharey(axes[row][0])

    diff_dfs = {mid: build_judge_diff_df(lie_dfs[mid], ext_lie_dfs[mid]) for mid in models}
    agree_dfs = {mid: build_judge_agreement_df(lie_dfs[mid], ext_lie_dfs[mid]) for mid in models}

    diff_abs_max = max(diff_dfs[mid].abs().max().max() for mid in models)
    diff_abs_max = max(diff_abs_max, 0.1)

    for col, model_id in enumerate(models):
        ax_diff = axes[0][col]
        sns.heatmap(
            diff_dfs[model_id].astype(float), ax=ax_diff, annot=True, fmt="+.2f",
            cmap="RdBu_r", vmin=-diff_abs_max, vmax=diff_abs_max,
            linewidths=0.5,
            cbar=False,
        )
        ax_diff.set_title(model_id, pad=12)
        ax_diff.set_xlabel("")
        ax_diff.set_ylabel("Difference" if col == 0 else "")

        ax_agree = axes[1][col]
        sns.heatmap(
            agree_dfs[model_id].astype(float), ax=ax_agree, annot=True, fmt=".0f",
            cmap="RdYlGn", vmin=0.0, vmax=1.0,
            linewidths=0.5,
            cbar=False,
        )
        ax_agree.set_xlabel("Subject prompt level")
        ax_agree.set_ylabel("Agreement" if col == 0 else "")

    sm_diff = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-diff_abs_max, vmax=diff_abs_max))
    sm_diff.set_array([])
    fig.colorbar(sm_diff, cax=cbar_axes[0], label="External − self (false claims)")

    sm_agree = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm_agree.set_array([])
    fig.colorbar(sm_agree, cax=cbar_axes[1], label=f"Direction agreement (threshold={_AGREEMENT_THRESHOLD})")

    for col in range(1, n):
        axes[0][col].set_yticklabels([])
        axes[1][col].set_yticklabels([])
    for col in range(n):
        axes[0][col].set_xticklabels([])

    judge_label = f" ({judge_model_id})" if judge_model_id else ""
    fig.suptitle(f"Judge agreement — self-eval vs. external judge{judge_label}", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "judge_agreement_all_models.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_judge_agreement_md_all_models(
    lie_dfs: Dict[str, pd.DataFrame],
    ext_lie_dfs: Dict[str, pd.DataFrame],
    charts_path: str,
    judge_model_id: str = None,
):
    subdir = os.path.join(charts_path, "judge_agreement")
    os.makedirs(subdir, exist_ok=True)

    judge_label = f" ({judge_model_id})" if judge_model_id else ""
    lines = [f"# Judge agreement — self-eval vs. external judge{judge_label}", ""]

    for model_id in lie_dfs:
        diff_df = build_judge_diff_df(lie_dfs[model_id], ext_lie_dfs[model_id])
        agree_df = build_judge_agreement_df(lie_dfs[model_id], ext_lie_dfs[model_id])
        lines += [
            f"## {model_id}", "",
            "### Signed difference (external − self)",
            "", diff_df.astype(float).to_markdown(floatfmt="+.2f"), "",
            f"### Directional agreement (threshold={_AGREEMENT_THRESHOLD})",
            "", agree_df.astype(float).to_markdown(floatfmt=".0f"), "",
        ]

    with open(os.path.join(subdir, "judge_agreement_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
