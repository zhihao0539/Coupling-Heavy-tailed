#!/usr/bin/env python3
"""Plot the 1,000-replicate comparison using the original edited figure style.

Adapted from original_plot_format.py. Censored paths remain in the survival
curve and are excluded from the conditional finite-time histogram.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHODS = [
    ("stereographic", "Stereographic", "tab:blue"),
    ("euclidean", "Euclidean", "tab:orange"),
    ("scs", "Sub-Cauchy (VI)", "tab:green"),
    ("dcs_bw", "DCS-Ball walk (VI)", "tab:purple"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "meeting_times.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "meeting_times_vi_1000.pdf"
        ),
    )
    parser.add_argument("--histogram-bins", type=int, default=34)
    parser.add_argument("--curve-points", type=int, default=1600)
    args = parser.parse_args()

    results = pd.read_csv(args.input)
    results["met"] = results["met"].astype(str).str.lower().eq("true")

    meeting_times: dict[str, np.ndarray] = {}
    finite_times: dict[str, np.ndarray] = {}
    acceptances: dict[str, float] = {}
    meeting_counts: dict[str, tuple[int, int]] = {}
    for method, _, _ in METHODS:
        selected = results.loc[results["method"].eq(method)]
        # A censored pair is represented by infinity in the survival curve.
        meeting_times[method] = np.where(
            selected["met"].to_numpy(dtype=bool),
            selected["meeting_time"].to_numpy(dtype=float),
            np.inf,
        )
        finite_times[method] = meeting_times[method][
            np.isfinite(meeting_times[method])
        ]
        acceptances[method] = float(
            selected[
                ["acceptance_rate_chain1", "acceptance_rate_chain2"]
            ].to_numpy(dtype=float).mean()
        )
        meeting_counts[method] = (
            int(selected["met"].sum()),
            int(len(selected)),
        )

    all_finite = np.concatenate(list(finite_times.values()))
    histogram_lower = max(1.0, float(np.min(all_finite)))
    histogram_upper = float(np.max(all_finite)) * 1.001
    bins = np.geomspace(
        histogram_lower, histogram_upper, args.histogram_bins + 1
    )

    # Include the full censoring horizon so non-meeting fractions appear as
    # plateaus at the right edge of the empirical survival curve.
    max_iteration = int(results["iterations"].max())
    positive_iters = np.unique(
        np.geomspace(1, max_iteration, args.curve_points).astype(int)
    )
    iters = np.concatenate([np.asarray([0], dtype=int), positive_iters])
    survival = {
        method: np.asarray(
            [np.mean(values > iteration) for iteration in iters], dtype=float
        )
        for method, values in meeting_times.items()
    }

    # sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method, label, color in METHODS:
        met, total = meeting_counts[method]
        sns.histplot(
            finite_times[method],
            bins=bins,
            stat="probability",
            alpha=0.7,
            color=color,
            label=(
                f"{label}, met={met}/{total}, "
                f"acc={acceptances[method]:.3f}"
            ),
            ax=axes[0],
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Meeting time")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Finite meeting times (met pairs only)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    # axes[0].grid(True, which="minor", axis="x", alpha=0.10)

    for method, label, color in METHODS:
        axes[1].plot(
            iters,
            survival[method],
            color=color,
            linewidth=1.8,
            label=label,
        )

    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlim(left=0.0, right=max_iteration)
    axes[1].set_ylim(-0.015, 1.015)
    axes[1].set_xlabel("Iteration $t$")
    axes[1].set_ylabel(
        r"Empirical survival $\widehat{\mathbb{P}}(\tau > t)$"
    )
    axes[1].set_title("Meeting-time survival")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    # axes[1].grid(True, which="minor", axis="x", alpha=0.10)

    # for axis in axes:
    #     axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
