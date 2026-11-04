"""Step 1: set the four VI maps and fit them once per dimension."""
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
REPOSITORY = FOLDER.parents[1]
sys.path.insert(0, str(REPOSITORY))

from target import DIMENSIONS, make_log_density, target_info
from generic_algorithms import run_stereographic_vi, run_scs_vi, run_dcs_matrix_vi

RESULTS = FOLDER / "results"
SEED = 20261027
VI_SETTINGS = dict(
    steps=20000, batch_size=2048, holdout_size=20000,
    average_last=500, learning_rates=(0.01, 0.002),
    change_after=10000, chunk_size=250,
)


def fit_dimension(d):
    target = make_log_density(d, jax=True)
    path = RESULTS / f"d_{d:04d}" / "fitted_parameters.json"
    run_stereographic_vi(
        target, d, mu=np.zeros(d), preconditioning_matrix=np.sqrt(d) * np.eye(d),
        seed=SEED + 1000*d, output_json=path, target_info=target_info(d), **VI_SETTINGS)
    run_scs_vi(
        target, d, mu=np.zeros(d), preconditioning_matrix=(np.sqrt(d)/11) * np.eye(d),
        horizontal_observer=np.zeros(d), latitude=1.1, label="SCS11",
        seed=SEED + 1000*d + 1, output_json=path, target_info=target_info(d), **VI_SETTINGS)
    run_scs_vi(
        target, d, mu=np.zeros(d), preconditioning_matrix=(9*np.sqrt(d)/19) * np.eye(d),
        horizontal_observer=np.zeros(d), latitude=1.9, label="SCS19",
        seed=SEED + 1000*d + 2, output_json=path, target_info=target_info(d), **VI_SETTINGS)
    run_dcs_matrix_vi(
        target, d, mu=np.zeros(d), preconditioning_matrix=np.eye(d),
        seed=SEED + 1000*d + 3, output_json=path, target_info=target_info(d), **VI_SETTINGS)


def main():
    for d in DIMENSIONS:
        fit_dimension(d)


if __name__ == '__main__':
    main()
