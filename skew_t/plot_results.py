"""Plot saved meeting times without running the samplers."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change to "new_meeting_times.csv" to plot a new run.
RESULTS_FILE = "meeting_times.csv"
FOLDER = Path(__file__).resolve().parent

def plot_results(results, output=FOLDER / "meeting_times.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    finite = results.loc[results.meeting_time > 0, "meeting_time"].to_numpy()
    cutoff = int(results["cutoff"].max())
    lower = max(1, finite.min()) if len(finite) else 1
    upper = max(lower * 1.01, finite.max() * 1.001) if len(finite) else max(cutoff, 2)
    bins = np.geomspace(lower, upper, 35)
    iters = np.r_[0, np.unique(np.geomspace(1, cutoff, 1600).astype(int))]
    methods = [("stereographic", "Stereographic", "tab:blue"),
               ("euclidean", "Euclidean", "tab:orange"),
               ("scs", "Sub-Cauchy (VI)", "tab:green"),
               ("dcs_bw", "DCS-Ball walk (VI)", "tab:purple")]
    for method, label, color in methods:
        selected = results[results.method == method]
        tau = np.where(selected.meeting_time > 0, selected.meeting_time, np.inf)
        finite_tau = tau[np.isfinite(tau)]
        acceptance = selected[["acceptance_rate_chain1", "acceptance_rate_chain2"]].to_numpy().mean()
        if len(finite_tau):
            axes[0].hist(finite_tau, bins=bins, weights=np.ones(len(finite_tau)) / len(finite_tau),
                         color=color, alpha=0.7, edgecolor="black", linewidth=0.5,
                         label=f"{label}, met={len(finite_tau)}/{len(tau)}, acc={acceptance:.3f}")
        survival = 1 - np.searchsorted(np.sort(tau), iters, side="right") / len(tau)
        axes[1].plot(iters, survival, color=color, linewidth=1.8, label=label)
        print(f"{label}: met {len(finite_tau)}/{len(tau)}, mean min(tau, cutoff)={np.minimum(tau, cutoff).mean():.3f}, acc={acceptance:.3%}")
    axes[0].set(xscale="log", xlabel="Meeting time", ylabel="Probability", title="Finite meeting times (met pairs only)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set(xlim=(0, cutoff), ylim=(-0.015, 1.015), xlabel="Iteration $t$",
                ylabel=r"Empirical survival $\widehat{\mathbb{P}}(\tau > t)$", title="Meeting-time survival")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results = pd.read_csv(FOLDER / RESULTS_FILE)
    plot_results(results, (FOLDER / RESULTS_FILE).with_suffix(".pdf"))
