"""Step 3: summarize and plot the five configured coupling methods."""
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
    "dcs": ("DCS ball walk (Mobius)", "tab:purple"),
    "euclidean": ("Euclidean", "tab:orange"),
}
MARKERS = {"stereographic": "o", "SCS11": "^", "SCS19": "s", "dcs": "D", "euclidean": "v"}


def read_results(root):
    path = root / "meeting_times.csv"
    if path.exists():
        return pd.read_csv(path)
    frames = [pd.read_csv(path).assign(dimension=d) for d in DIMENSIONS
              if (path := root / f"d_{d:04d}" / "meeting_times.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["method"])


def load_results():
    local = read_results(RESULTS)
    local = local[local.method.isin(STYLES)].copy()
    if local.empty:
        raise FileNotFoundError(f"No meeting-time results in {RESULTS}; run run_experiment.py first.")
    if "mu_fixed" not in local or not local.mu_fixed.eq(True).all():
        raise ValueError(
            "Saved results do not use mu=0 for every map. "
            "Refit if needed (python run_vi.py), then rerun run_experiment.py.")
    scs_rows = local[local.method.isin(("SCS11", "SCS19"))]
    fixed_columns = ["scs_mu_fixed", "scs_observer_fixed"]
    if not set(fixed_columns).issubset(local.columns) or not scs_rows[fixed_columns].eq(True).all().all():
        raise ValueError(
            "Saved SCS results do not use the fixed-zero location and observer. "
            "Refit if needed (python run_vi.py), "
            "then rerun run_experiment.py.")
    dcs_rows = local[local.method == "dcs"]
    if "dcs_map" not in local or not dcs_rows.dcs_map.eq("mobius").all():
        raise ValueError(
            "Saved DCS results do not use the Mobius map. "
            "Refit if needed, then rerun run_experiment.py.")
    frames = []
    for name in STYLES:
        rows = local[local.method == name]
        if rows.empty:
            raise FileNotFoundError(f"No saved results for {name} in {RESULTS}.")
        rows = rows.copy().assign(method=name)
        if "h" not in rows:
            rows["h"] = [STEP_SIZES[name][list(DIMENSIONS).index(int(d))] for d in rows.dimension]
        for column, value in {"cutoff": MAX_STEPS, "initialization": "target",
                              "scs_latitude": {"SCS11": 1.1, "SCS19": 1.9}.get(name, np.nan)}.items():
            if column not in rows:
                rows[column] = value
        rows["df"] = 2
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
    horizon = int(results.cutoff.iloc[0]) if horizon is None else int(horizon)
    summaries = []
    for (d, method), rows in results.groupby(["dimension", "method"], sort=True):
        tau = rows.meeting_time.to_numpy()
        met = tau >= 0
        met_tau = tau[met]
        met_se = met_tau.std(ddof=1)/np.sqrt(len(met_tau)) if len(met_tau) > 1 else np.nan
        restricted = np.where(met, tau, horizon)
        se = restricted.std(ddof=1)/np.sqrt(len(tau)) if len(tau) > 1 else np.nan
        summaries.append(dict(
            dimension=int(d), df=2, method=method, h=float(rows.h.iloc[0]),
            scs_latitude=float(rows.scs_latitude.iloc[0]), cutoff=horizon,
            n_replicates=len(tau), n_met=int(met.sum()), n_censored=int((~met).sum()),
            fraction_met=float(met.mean()),
            mean_meeting_time=float(tau.mean()) if met.all() else np.nan,
            mean_meeting_time_mcse=float(se) if met.all() else np.nan,
            mean_met_pairs=float(met_tau.mean()) if len(met_tau) else np.nan,
            mean_met_pairs_mcse=float(met_se),
            restricted_mean=float(restricted.mean()), restricted_mean_mcse=float(se),
            acceptance_rate=rows[["acceptance_rate_chain1", "acceptance_rate_chain2"]].mean().mean(),
        ))
    return pd.DataFrame(summaries)


def main():
    results = load_results()
    summary = summarize(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17.25, 5))
    for name, (label, color) in STYLES.items():
        data = summary[summary.method == name].sort_values("dimension")
        if data.empty:
            continue
        style = dict(marker=MARKERS[name], linestyle="-", markerfacecolor="none", color=color, label=label)
        axes[0].errorbar(data.dimension, data.mean_meeting_time,
                        yerr=data.mean_meeting_time_mcse, capsize=3, **style)
        axes[1].errorbar(data.dimension, data.mean_met_pairs,
                        yerr=data.mean_met_pairs_mcse, capsize=3, **style)
        axes[2].plot(data.dimension, data.h, **style)
    axes[0].set(title="Mean (only when all pairs met)", ylabel="Mean meeting time")
    axes[1].set(title="Mean (pairs that met)", ylabel="Mean meeting time")
    axes[2].set(title="Step size versus dimension", ylabel="Step size h")
    dimensions = sorted(summary.dimension.unique())
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(dimensions, [str(d) for d in dimensions])
        ax.set_xlabel("Dimension d (multivariate Student-t, df = 2)")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    axes[2].legend(fontsize=9)
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
