"""Plot meeting times from the shared runner without rerunning the samplers."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(results, output):
    """Plot a result DataFrame from experiments with a common iteration cutoff.

    Nonnegative meeting times include meetings at initialization; -1 denotes
    censoring. Histograms use meeting pairs only, while survival uses all pairs.
    """
    if results.empty:
        raise ValueError("Supply at least one result row.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    finite = results.loc[results.meeting_time >= 0, "meeting_time"].to_numpy()
    horizon = int(results.iterations.max())
    bins = np.linspace(finite.min(), max(finite.max(), finite.min() + 1), 51) if len(finite) else np.linspace(0, max(horizon, 1), 51)
    iters = np.arange(horizon + 1)
    methods = [("stereographic", "Stereographic", "tab:blue"),
               ("euclidean", "Euclidean", "tab:orange"),
               ("scs", "Sub-Cauchy", "tab:green"),
               ("dcs", "Diffeomorphism", "tab:purple")]
    for method, label, color in methods:
        selected = results[results.method == method]
        if selected.empty:
            continue
        tau = np.where(selected.meeting_time >= 0, selected.meeting_time, np.inf)
        finite_tau = tau[np.isfinite(tau)]
        rates = selected[["acceptance_rate_chain1", "acceptance_rate_chain2"]].to_numpy()
        rates = rates[np.isfinite(rates)]
        acceptance = rates.mean() if len(rates) else np.nan
        if len(finite_tau):
            axes[0].hist(finite_tau, bins=bins, weights=np.ones(len(finite_tau)) / len(finite_tau),
                         alpha=0.7, color=color, edgecolor="black", linewidth=0.5,
                         label=f"{label}, acc={acceptance:.3f}")
        survival = 1 - np.searchsorted(np.sort(tau), iters, side="right") / len(tau)
        axes[1].plot(iters, survival, color=color, linewidth=1.5, label=label)
        print(f"{label}: met {len(finite_tau)}/{len(tau)}, mean min(tau, cutoff)={np.minimum(tau, horizon).mean():.3f}, acc={acceptance:.3%}")
    axes[0].set(xlabel="Meeting time", ylabel="Probability", title="Distribution of meeting times")
    axes[1].set(xlabel="Iteration $t$", ylabel=r"$\widehat{\mathbb{P}}(\tau > t)$", title="Meeting-time survival")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Meeting-time CSV from the shared runner")
    parser.add_argument("--output", type=Path, help="Output figure path; default: CSV path with .pdf suffix")
    args = parser.parse_args()
    plot_results(pd.read_csv(args.csv), args.output or args.csv.with_suffix(".pdf"))
