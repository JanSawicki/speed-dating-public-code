import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._common import TIERS, _model_slug


def _normalised_lift(total: float, rate_l0: float) -> float:
    """Normalise lift by the available headroom in the direction of the lift.

    Positive lift: headroom = 1 - rate_L0  (how far the rate can rise)
    Negative lift: headroom = rate_L0       (how far the rate can fall)
    Returns NaN when the headroom in the direction of movement is zero.
    """
    if total >= 0:
        headroom = 1.0 - rate_l0
    else:
        headroom = rate_l0
    if headroom == 0.0:
        return float("nan")
    return total / headroom


def build_deception_lift_df(ranking_df: pd.DataFrame) -> pd.DataFrame:
    df = ranking_df.astype(float)
    levels = [c for c in df.columns]
    result = {col: df[col] for col in levels}
    for i in range(1, len(levels)):
        prev, curr = levels[i - 1], levels[i]
        n_prev = prev.split()[-1]
        n_curr = curr.split()[-1]
        result[f"L{n_prev}→L{n_curr}"] = df[curr] - df[prev]
    total = df[levels[-1]] - df[levels[0]]
    result["Total"] = total
    result["Norm. Total"] = pd.Series(
        {tier: _normalised_lift(total[tier], df.loc[tier, levels[0]]) for tier in TIERS}
    )
    return pd.DataFrame(result, index=TIERS)


def save_deception_lift_md_all_models(lift_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "deception_lift")
    os.makedirs(subdir, exist_ok=True)
    lines = ["# Deception lift — all models", ""]
    lines += [
        "Norm. Total = Total / max_possible_lift, where max_possible_lift = (1 − rate_L0) "
        "for positive lift and rate_L0 for negative lift. NaN = no headroom in that direction.",
        "",
    ]
    for model_id, lift_df in lift_dfs.items():
        lines += [f"## {model_id}", "", lift_df.to_markdown(floatfmt=".2f"), ""]
    with open(os.path.join(subdir, "deception_lift_all_models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_deception_lift_chart_all_models(lift_dfs: Dict[str, pd.DataFrame], charts_path: str):
    subdir = os.path.join(charts_path, "deception_lift")
    os.makedirs(subdir, exist_ok=True)
    models = list(lift_dfs.keys())
    n_models = len(models)
    x = np.arange(len(TIERS))
    width = 0.8 / n_models

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(n_models * 5.0, 3.84))

    for i, model_id in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * width
        lifts = [lift_dfs[model_id].loc[tier, "Total"] for tier in TIERS]
        norm_lifts = [lift_dfs[model_id].loc[tier, "Norm. Total"] for tier in TIERS]
        ax_raw.bar(x + offset, lifts, width, label=model_id)
        # plot NaN as zero-height bar so spacing is consistent
        norm_vals = [v if not np.isnan(v) else 0.0 for v in norm_lifts]
        bars = ax_norm.bar(x + offset, norm_vals, width, label=model_id)
        # hatch NaN bars to signal undefined
        for bar, v in zip(bars, norm_lifts):
            if np.isnan(v):
                bar.set_hatch("///")
                bar.set_edgecolor("grey")

    for ax, title, ylabel in [
        (ax_raw, "Raw deception lift (Total = L3 − L0)", "Raw lift"),
        (ax_norm, "Normalised deception lift (Total / headroom)", "Normalised lift"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(TIERS)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.legend(fontsize=7)

    ax_norm.text(
        0.5, -0.14, "Hatched bars = undefined (no headroom in lift direction)",
        transform=ax_norm.transAxes, ha="center", fontsize=7, color="grey",
    )

    plt.tight_layout()
    fig.savefig(os.path.join(subdir, "deception_lift_all_models.pdf"), format="pdf")
    plt.close(fig)
