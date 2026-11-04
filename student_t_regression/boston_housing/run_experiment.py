"""Run the fixed-parameter Student-t regression comparison."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent.parent))

from generic_algorithms import (
    run_stereographic_experiment, run_euclidean_experiment, combine_results, save_results,
)
from generic_algorithms.plot_results import plot_results
from student_t_regression.boston_housing.target import load_data, log_density

N_REP = 1000
N_STEPS = 20000
SEED = 42
PROGRESS_EVERY = 25
STEREOGRAPHIC_H = 0.007        # Tangent-Gaussian scale on the sphere.
EUCLIDEAN_H = 0.03            # Euclidean Gaussian standard deviation.
OUTPUT = FOLDER / 'meeting_times.csv'


def generate_initial_pairs(dimension, n_pairs=N_REP, seed=SEED):
    """Draw the common N(0, 100) starting pairs using the original random stream."""
    return np.random.RandomState(seed).normal(0, 10, (n_pairs, 2, dimension))


def run_stereographic(starts, target):
    dimension = starts.shape[-1]
    return run_stereographic_experiment(
        target, starts, STEREOGRAPHIC_H, N_STEPS,
        mu=np.zeros(dimension), preconditioning_matrix=np.sqrt(dimension) * np.eye(dimension),
        seed=SEED, seed_offset=100_000,
        stop_at_meeting=False, progress_every=PROGRESS_EVERY)


def run_euclidean(starts, target):
    return run_euclidean_experiment(
        target, starts, EUCLIDEAN_H, N_STEPS,
        seed=SEED, seed_offset=200_000,
        stop_at_meeting=False, progress_every=PROGRESS_EVERY)


def main():
    X, y = load_data()
    starts = generate_initial_pairs(X.shape[1] + 1, N_REP, SEED)

    def target(theta):
        return log_density(theta, X, y)

    rows = combine_results(
        run_stereographic(starts, target),
        run_euclidean(starts, target),
    )
    save_results(rows, OUTPUT)
    plot_results(pd.DataFrame(rows), OUTPUT.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
