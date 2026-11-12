"""Plot the d=50 UPAVE estimates with one-MCSE error bars, without rerunning MCMC."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results/d_0050/asymptotic_variance"
METHODS = {
    "stereographic": ("Stereographic", "tab:blue"),
    "dcs": ("DCS (matrix)", "tab:purple"),
    "euclidean": ("Euclidean", "tab:orange"),
}


def load_estimates():
    rows = []
    for method, (label, color) in METHODS.items():
        data = pd.read_csv(RESULTS/f"{method}.csv")
        if len(data)<2 or data.replicate.duplicated().any() or not data.method.eq(method).all():
            raise ValueError(f"{method}: supply at least two distinct estimator replicates.")
        if not data.dimension.eq(50).all() or not data.observable.eq("x1").all():
            raise ValueError("This plot requires d=50 and h(x)=x_1.")
        if not data.initial_distribution.eq("N(0,I)").all():
            raise ValueError("This comparison requires standard-Gaussian initial states.")
        values = data.v.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{method}: all estimator replicates must be finite.")
        rows.append(dict(method=method, label=label, color=color, n_replicates=len(values),
                         mean_v=float(values.mean()), mcse=float(values.std(ddof=1)/np.sqrt(len(values)))))
    return pd.DataFrame(rows)


def main():
    data = load_estimates()
    mean, error = data.mean_v.to_numpy(), data.mcse.to_numpy()
    if np.any(mean-error<=0):
        raise ValueError("A logarithmic plot requires positive mean-minus-MCSE values.")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = np.arange(len(data))
    ax.bar(x, mean, width=.6, yerr=error, capsize=5, color=data.color,
           edgecolor="black", linewidth=.6,
           error_kw=dict(ecolor="black", elinewidth=1.3, capthick=1.3))
    ax.set_yscale("log")
    ax.set_ylim(.8, 2*float(np.max(mean+error)))
    ax.set_xticks(x, data.label)
    ax.set_xlabel("Method")
    ax.set_ylabel(r"Estimated asymptotic variance $\widehat{v}(P,h)$"+"\n(log scale)")
    for position, value, se in zip(x, mean, error):
        ax.text(position, (value+se)*1.13, f"{value:,.2f} ± {se:,.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", which="major", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS/f"asymptotic_variance_comparison.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(data[["method","n_replicates","mean_v","mcse"]].to_string(index=False))


if __name__ == "__main__":
    main()
