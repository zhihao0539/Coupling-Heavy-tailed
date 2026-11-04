"""Step 3: summarize and plot the six configured coupling methods."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_vi import RESULTS
from run_experiment import STEP_SIZES, MAX_STEPS
from target import DIMENSIONS
from generic_algorithms.plot_results import plot_results as plot_generic

STYLES = {
    "stereographic": ("Stereographic", "tab:blue"),
    "SCS11": ("SCS11", "tab:green"),
    "SCS19": ("SCS19", "tab:red"),
    "dcs": ("DCS (matrix)", "tab:purple"),
    "euclidean": ("Euclidean", "tab:orange"),
    "dcs_rw": ("DCS-Gaussian random walk (matrix)", "tab:brown"),
}
MARKERS = {"stereographic": "o", "SCS11": "^", "SCS19": "s", "dcs": "D", "euclidean": "v", "dcs_rw": "P"}


def read_results(root):
    path = root / "meeting_times.csv"
    if path.exists():
        return pd.read_csv(path)
    frames = [pd.read_csv(path).assign(dimension=d) for d in DIMENSIONS
              if (path := root / f"d_{d:04d}" / "meeting_times.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["method"])


def load_results():
    local = read_results(RESULTS)
    frames = []
    for name in STYLES:
        rows = local[local.method == name]
        if rows.empty:
            if name == "dcs_rw":
                print("No saved DCS random-walk results yet; plotting the existing methods.")
                continue
            raise FileNotFoundError(f"No saved results for {name} in {RESULTS}.")
        rows = rows.copy().assign(method=name)
        if "h" not in rows:
            rows["h"] = [STEP_SIZES[name][list(DIMENSIONS).index(int(d))] for d in rows.dimension]
        for column, value in {"cutoff": MAX_STEPS, "initialization": "target",
                              "scs_latitude": {"SCS11": 1.1, "SCS19": 1.9}.get(name, np.nan)}.items():
            if column not in rows:
                rows[column] = value
        rows["df"] = rows.dimension + 1
        frames.append(rows)
    results = pd.concat(frames, ignore_index=True)
    if results.duplicated(["dimension", "method", "replicate"]).any():
        raise ValueError("Duplicate method/dimension/replicate results.")
    if not results.initialization.eq("target").all():
        raise ValueError("This comparison requires target-distributed initial states.")
    return results


def summarize(results, horizon=None):
    if results.cutoff.nunique() != 1:
        raise ValueError("The saved runs must use a common cutoff.")
    cutoff = int(results.cutoff.iloc[0])
    horizon = cutoff if horizon is None else int(horizon)
    if not 0 < horizon <= cutoff:
        raise ValueError("horizon must be positive and no larger than the saved cutoff.")
    summaries = []
    for (d, method), rows in results.groupby(["dimension", "method"], sort=True):
        tau = rows.meeting_time.to_numpy()
        met = (tau >= 0) & (tau <= horizon)
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
        ax.set_xlabel("Dimension d (multivariate Student-t, df = d + 1)")
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
