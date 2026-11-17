"""Fit NBA Gaussian RWMH preconditioning, preserving the saved SRWM/DCS fits."""
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent.parent))

from generic_algorithms import run_euclidean_vi, run_stereographic_vi, run_dcs_matrix_vi
from student_t_regression.nba_salary.target import D, NU, PRIOR_DF, RESPONSE, log_density_jax

PARAMETERS_FILE = FOLDER / 'fitted_parameters.json'
STEPS = 4000
BATCH_SIZE = 2048
HOLDOUT_SIZE = 20000
AVERAGE_LAST = 500
LEARNING_RATES = (0.01, 0.002)
CHANGE_AFTER = 3000
CHUNK_SIZE = 250
VI_SEED = 26092801
DCS_VI_SEED = VI_SEED + 1
EUCLIDEAN_VI_SEED = VI_SEED + 2
DCS_INITIAL_BETA = 2.0
TARGET_INFO = {
    'response': RESPONSE, 'error_df': NU,
    'scale_coordinate': 'v=sinh(log_sigma)',
    'log_sigma_prior': 'flat',
    'beta_prior': 'product_standard_student_t',
    'beta_prior_df': PRIOR_DF,
}


def fit_euclidean():
    """Fit both the location and full matrix using standard Gaussian draws."""
    return run_euclidean_vi(
        log_density_jax, D, mu=np.full(D, .25), preconditioning_matrix=np.eye(D),
        tune=('mu', 'preconditioning_matrix'),
        seed=EUCLIDEAN_VI_SEED, chunk_size=CHUNK_SIZE, output_json=PARAMETERS_FILE,
        target_info=TARGET_INFO,
        steps=STEPS, batch_size=BATCH_SIZE, holdout_size=HOLDOUT_SIZE,
        average_last=AVERAGE_LAST, learning_rates=LEARNING_RATES,
        change_after=CHANGE_AFTER)


def fit_stereographic():
    """Fit the center and full Cholesky factor from uniform sphere points."""
    return run_stereographic_vi(
        log_density_jax, D, mu=np.zeros(D), preconditioning_matrix=np.sqrt(D) * np.eye(D),
        seed=VI_SEED, chunk_size=CHUNK_SIZE, output_json=PARAMETERS_FILE,
        target_info=TARGET_INFO,
        steps=STEPS, batch_size=BATCH_SIZE, holdout_size=HOLDOUT_SIZE,
        average_last=AVERAGE_LAST, learning_rates=LEARNING_RATES,
        change_after=CHANGE_AFTER)


def fit_dcs():
    """Tune the location, full matrix, and beta using uniform unit-ball samples."""
    return run_dcs_matrix_vi(
        log_density_jax, D, mu=np.zeros(D),
        preconditioning_matrix=np.sqrt(D) * np.eye(D), beta=DCS_INITIAL_BETA,
        tune=('mu', 'preconditioning_matrix', 'beta'),
        seed=DCS_VI_SEED, chunk_size=CHUNK_SIZE, output_json=PARAMETERS_FILE,
        target_info=TARGET_INFO,
        steps=STEPS, batch_size=BATCH_SIZE, holdout_size=HOLDOUT_SIZE,
        average_last=AVERAGE_LAST, learning_rates=LEARNING_RATES,
        change_after=CHANGE_AFTER)


def main():
    fit_euclidean()


if __name__ == '__main__':
    main()
