"""Run the fixed-parameter Student-t regression comparison."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import default_parameters, run_experiment
from student_t_regression.target import load_data, log_density
from student_t_regression.plot_results import plot_results

N_REP = 1000
N_STEPS = 20000
SEED = 42


if __name__ == '__main__':
    X, y = load_data()
    d = X.shape[1] + 1
    initial_pairs = np.random.RandomState(SEED).normal(0, 10, (N_REP, 2, d))
    methods = {
        'stereographic': {'h': .007, 'parameters': default_parameters(d, 'stereographic', radius=np.sqrt(d)), 'seed_offset': 100000},
        'euclidean': {'h': .03, 'seed_offset': 200000},
    }
    output = FOLDER / 'meeting_times.csv'
    rows = run_experiment(lambda theta: log_density(theta, X, y), initial_pairs,
                          methods, N_STEPS, seed=SEED, output_csv=output)
    plot_results(pd.DataFrame(rows), output.with_suffix('.pdf'))
