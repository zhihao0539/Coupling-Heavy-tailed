"""Fit the maps, run the five couplings with configured steps, then plot."""


def main():
    from run_vi import main as fit
    from run_experiment import main as simulate
    from plot_results import main as plot

    fit()
    simulate()
    plot()


if __name__ == "__main__":
    main()
