"""Fit the VI maps, run all six coupling methods, then plot.

For existing results only, run plot_results.py instead.
"""


def main():
    from run_vi import main as fit
    from run_experiment import main as simulate
    from plot_results import main as plot

    fit()
    simulate()
    plot()


if __name__ == "__main__":
    main()
