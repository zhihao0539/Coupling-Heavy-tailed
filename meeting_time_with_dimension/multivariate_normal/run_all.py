"""Run the five fixed-map normal coupling methods, then plot (no VI).

For existing results only, run plot_results.py instead.
"""


def main():
    from run_experiment import main as simulate
    from plot_results import main as plot

    simulate()
    plot()


if __name__ == "__main__":
    main()
