"""Summarize and plot meeting times and calibrated step sizes for the five couplings."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_experiment import RESULTS
from target import DIMENSIONS
from generic_algorithms.plot_results import plot_results as plot_generic

STYLES = {
    "stereographic": ("Stereographic", "tab:blue"),
    "SCS11": ("SCS11", "tab:green"),
    "SCS19": ("SCS19", "tab:red"),
    "dcs": ("DCS matrix ball walk", "tab:purple"),
    "euclidean": ("Euclidean", "tab:orange"),
}
MARKERS = {"stereographic": "o", "SCS11": "^", "SCS19": "s", "dcs": "D", "euclidean": "v"}


def read_results(root):
    frames = [pd.read_csv(path) for d in DIMENSIONS
              if (path := root / f"d_{d:04d}" / "meeting_times.csv").exists()]
    if frames:
        return pd.concat(frames, ignore_index=True)
    path = root / "meeting_times.csv"
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"No meeting-time results in {root}. Run run_experiment.py first.")


def load_results():
    results = read_results(RESULTS)
    for d, rows in results.groupby("dimension"):
        if set(rows.method) != set(STYLES):
            raise ValueError(f"Incomplete set of coupling methods for dimension {d}.")
    if results.duplicated(["dimension", "method", "replicate"]).any():
        raise ValueError("Duplicate method/dimension/replicate results.")
    if not results.initialization.eq("target").all():
        raise ValueError("This comparison requires target-distributed initial states.")
    return results


def summarize(results, horizon=None):
    if results.cutoff.nunique() != 1:
        raise ValueError("The saved runs must use a common cutoff.")
    horizon = int(results.cutoff.iloc[0]) if horizon is None else int(horizon)
    summaries = []
    for (d, method), rows in results.groupby(["dimension", "method"], sort=True):
        tau = rows.meeting_time.to_numpy()
        met = tau >= 0
        restricted = np.where(met, tau, horizon)
        se = restricted.std(ddof=1)/np.sqrt(len(tau)) if len(tau) > 1 else np.nan
        summaries.append(dict(
            dimension=int(d), df=int(d)+1, method=method, h=float(rows.h.iloc[0]),
            scs_latitude=float(rows.scs_latitude.iloc[0]), cutoff=horizon,
            n_replicates=len(tau), n_met=int(met.sum()), n_censored=int((~met).sum()),
            fraction_met=float(met.mean()),
            mean_meeting_time=float(tau.mean()) if met.all() else np.nan,
            mean_meeting_time_mcse=float(se) if met.all() else np.nan,
            restricted_mean=float(restricted.mean()), restricted_mean_mcse=float(se),
            acceptance_rate=rows[["acceptance_rate_chain1", "acceptance_rate_chain2"]].mean().mean(),
        ))
    return pd.DataFrame(summaries)


def main():
    results = load_results()
    summary = summarize(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for name, (label, color) in STYLES.items():
        data = summary[summary.method == name].sort_values("dimension")
        if data.empty:
            continue
        style = dict(marker=MARKERS[name], linestyle="-", markerfacecolor="none", color=color, label=label)
        axes[0].errorbar(data.dimension, data.mean_meeting_time,
                        yerr=data.mean_meeting_time_mcse, capsize=3, **style)
        axes[1].plot(data.dimension, data.h, **style)
    axes[0].set(title="Mean (only when all pairs met)", ylabel="Mean meeting time")
    axes[1].set(title="Step size versus dimension", ylabel="Step size h")
    dimensions = sorted(summary.dimension.unique())
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(dimensions, [str(d) for d in dimensions])
        ax.set_xlabel("Dimension d (each coordinate has df = d + 1)")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS / f"meeting_times_vs_dimension.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    for d, rows in results.groupby("dimension", sort=True):
        folder = RESULTS / f"d_{int(d):04d}"
        folder.mkdir(parents=True, exist_ok=True)
        summary[summary.dimension == d].to_csv(folder / "summary.csv", index=False)
        plot_generic(rows, folder / "meeting_times.pdf", log_scale=True,
                     horizon=int(results.cutoff.iloc[0]), styles=STYLES)


if __name__ == "__main__":
    main()
