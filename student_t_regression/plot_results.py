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
    horizon = int(results.iterations.max())
    bins = np.linspace(finite.min(), max(finite.max(), finite.min() + 1), 51) if len(finite) else np.linspace(0, horizon, 51)
    iters = np.arange(horizon + 1)
    for method, label, color in [("stereographic", "Stereographic", "tab:blue"),
                                  ("euclidean", "Euclidean", "tab:orange")]:
        selected = results[results.method == method]
        tau = np.where(selected.meeting_time > 0, selected.meeting_time, np.inf)
        finite_tau = tau[np.isfinite(tau)]
        acceptance = selected[["acceptance_rate_chain1", "acceptance_rate_chain2"]].to_numpy().mean()
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
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results = pd.read_csv(FOLDER / RESULTS_FILE)
    plot_results(results, (FOLDER / RESULTS_FILE).with_suffix(".pdf"))
