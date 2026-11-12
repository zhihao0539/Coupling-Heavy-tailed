"""Plot means where all pairs met and means among met pairs at every dimension."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_experiment import RESULTS, N_REP, MAX_STEPS
from target import DIMENSIONS, RHO
from generic_algorithms.plot_results import plot_results as plot_generic

STYLES = {
    "stereographic": ("Stereographic", "tab:blue"),
    "dcs": ("DCS ball walk (matrix)", "tab:purple"),
    "euclidean": ("Euclidean", "tab:orange"),
}
MARKERS = {"stereographic": "o", "dcs": "D", "euclidean": "v"}


def load_results():
    frames = [pd.read_csv(RESULTS/f"d_{d:04d}"/"meeting_times.csv") for d in DIMENSIONS]
    results = pd.concat(frames, ignore_index=True)
    if set(results.method) != set(STYLES) or results.duplicated(["dimension", "method", "replicate"]).any():
        raise ValueError("Results must contain exactly the three configured methods, with unique pairs.")
    if not results.target.eq("multivariate_normal_ar1").all() or not np.allclose(results.rho, RHO):
        raise ValueError("Results must use the configured AR(1) Gaussian target.")
    if not results.initialization.eq("target").all() or not results.cutoff.eq(MAX_STEPS).all():
        raise ValueError("Results must use stationary starts and the common cutoff.")
    expected = np.where(results.method.eq("euclidean"), "none", "reverse_kl")
    if not results.map_source.eq(expected).all():
        raise ValueError("Transformed methods must use the new reverse-KL fits.")
    dcs = results.loc[results.method.eq("dcs")]
    if not dcs.dcs_map.eq("matrix").all() or not np.isfinite(dcs.beta).all() or not dcs.beta.gt(2).all():
        raise ValueError("DCS must use the fitted matrix map with beta>2.")
    for d, rows in results.groupby("dimension"):
        if set(rows.method) != set(STYLES) or not rows.groupby("method").size().eq(N_REP).all():
            raise ValueError(f"Incomplete comparison for dimension {d}.")
    return results


def summarize(results):
    rows = []
    for (d, method), data in results.groupby(["dimension", "method"], sort=True):
        tau = data.meeting_time.to_numpy()
        met = tau >= 0
        met_tau = tau[met]
        met_mean = float(met_tau.mean()) if len(met_tau) else np.nan
        met_se = float(met_tau.std(ddof=1)/np.sqrt(len(met_tau))) if len(met_tau)>1 else np.nan
        restricted = np.where(met, tau, MAX_STEPS)
        se = restricted.std(ddof=1)/np.sqrt(len(tau))
        rows.append(dict(dimension=int(d), method=method, h=float(data.h.iloc[0]),
            cutoff=MAX_STEPS, n_replicates=len(tau), n_met=int(met.sum()),
            n_censored=int((~met).sum()), fraction_met=float(met.mean()),
            mean_meeting_time=met_mean if met.all() else np.nan,
            mean_meeting_time_mcse=met_se if met.all() else np.nan,
            mean_met_pairs=met_mean, mean_met_pairs_mcse=met_se,
            restricted_mean=float(restricted.mean()), restricted_mean_mcse=float(se),
            acceptance_rate=float(data[["acceptance_rate_chain1", "acceptance_rate_chain2"]].mean().mean())))
    return pd.DataFrame(rows)


def plot_dimension_summary(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for name, (label, color) in STYLES.items():
        data = summary.loc[summary.method.eq(name)].sort_values("dimension")
        style = dict(marker=MARKERS[name], linestyle="-", markerfacecolor="none", color=color, label=label)
        axes[0].errorbar(data.dimension, data.mean_meeting_time,
                        yerr=data.mean_meeting_time_mcse, capsize=3, **style)
        axes[1].errorbar(data.dimension, data.mean_met_pairs,
                        yerr=data.mean_met_pairs_mcse, capsize=3, **style)
    axes[0].set(title="Mean (only when all pairs met)", ylabel="Mean meeting time")
    axes[1].set(title="Mean among met pairs", ylabel="Mean meeting time")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(DIMENSIONS, [str(d) for d in DIMENSIONS])
        ax.set_xlabel("Dimension d")
        ax.grid(True, alpha=.3)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS/f"meeting_times_vs_dimension.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    results = load_results()
    summary = summarize(results)
    summary.to_csv(RESULTS/"summary.csv", index=False)
    plot_dimension_summary(summary)
    for d, rows in results.groupby("dimension", sort=True):
        folder = RESULTS/f"d_{int(d):04d}"
        summary.loc[summary.dimension.eq(d)].to_csv(folder/"summary.csv", index=False)
        plot_generic(rows, folder/"meeting_times.pdf", log_scale=True,
                     horizon=int(rows.iterations.max()), styles=STYLES)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
