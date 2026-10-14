"""Plot meeting times from the shared runner without rerunning the samplers."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(results, output, *, log_scale=False, horizon=None):
    """Plot a result DataFrame from experiments with a common iteration cutoff.

    Nonnegative meeting times include meetings at initialization; -1 denotes
    censoring. Histograms use meeting pairs only, while survival uses all pairs.
    ``log_scale=True`` uses logarithmic histogram bins and a symlog survival
    axis, for meeting times spanning several orders of magnitude; meetings at
    time 0 are then left out of the histogram. ``horizon`` is the common
    iteration cutoff, by default the largest recorded number of iterations.
    """
    if results.empty:
        raise ValueError("Supply at least one result row.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    finite = results.loc[results.meeting_time >= 0, "meeting_time"].to_numpy()
    horizon = int(results.iterations.max() if horizon is None else horizon)
    censored = bool((results.meeting_time < 0).any())
    if log_scale:
        positive = finite[finite > 0]
        lower = positive.min() if len(positive) else 1
        upper = max(lower * 1.01, positive.max() * 1.001) if len(positive) else max(horizon, 2)
        bins = np.geomspace(lower, upper, 35)
        iters = np.r_[0, np.unique(np.geomspace(1, max(horizon, 1), 1600).astype(int))]
    else:
        bins = np.linspace(finite.min(), max(finite.max(), finite.min() + 1), 51) if len(finite) else np.linspace(0, max(horizon, 1), 51)
        iters = np.arange(horizon + 1)
    methods = [("stereographic", "Stereographic", "tab:blue"),
               ("euclidean", "Euclidean", "tab:orange"),
               ("scs", "Sub-Cauchy", "tab:green"),
               ("dcs", "DCS-Ball walk", "tab:purple")]
    for method, label, color in methods:
        selected = results[results.method == method]
        if selected.empty:
            continue
        tau = np.where(selected.meeting_time >= 0, selected.meeting_time, np.inf)
        finite_tau = tau[np.isfinite(tau)]
        rates = selected[["acceptance_rate_chain1", "acceptance_rate_chain2"]].to_numpy()
        rates = rates[np.isfinite(rates)]
        acceptance = rates.mean() if len(rates) else np.nan
        shown = finite_tau[finite_tau > 0] if log_scale else finite_tau
        if len(shown):
            met = f", met={len(finite_tau)}/{len(tau)}" if censored else ""
            axes[0].hist(shown, bins=bins, weights=np.full(len(shown), 1 / len(finite_tau)),
                         alpha=0.7, color=color, edgecolor="black", linewidth=0.5,
                         label=f"{label}{met}, acc={acceptance:.3f}")
        survival = 1 - np.searchsorted(np.sort(tau), iters, side="right") / len(tau)
        axes[1].plot(iters, survival, color=color, linewidth=1.5, label=label)
        print(f"{label}: met {len(finite_tau)}/{len(tau)}, mean min(tau, cutoff)={np.minimum(tau, horizon).mean():.3f}, acc={acceptance:.3%}")
    axes[0].set(xlabel="Meeting time", ylabel="Probability", title="Distribution of meeting times")
    axes[1].set(xlabel="Iteration $t$", ylabel=r"$\widehat{\mathbb{P}}(\tau > t)$", title="Meeting-time survival")
    if log_scale:
        axes[0].set_xscale("log")
        axes[1].set_xscale("symlog", linthresh=1)
        axes[1].set(xlim=(0, horizon), ylim=(-0.015, 1.015))
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9 if log_scale else None)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Meeting-time CSV from the shared runner")
    parser.add_argument("--output", type=Path, help="Output figure path; default: CSV path with .pdf suffix")
    parser.add_argument("--log-scale", action="store_true",
                        help="Logarithmic meeting-time axes, for times spanning orders of magnitude")
    parser.add_argument("--horizon", type=int, help="Common iteration cutoff; default: largest recorded iterations")
    args = parser.parse_args()
    plot_results(pd.read_csv(args.csv), args.output or args.csv.with_suffix(".pdf"),
                 log_scale=args.log_scale, horizon=args.horizon)
