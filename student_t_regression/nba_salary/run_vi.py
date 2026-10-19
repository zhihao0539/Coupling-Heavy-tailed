"""Fit the NBA stereographic map using the shared VI runner."""
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent.parent))

from generic_algorithms import run_vi
from student_t_regression.nba_salary.target import D, NU, PRIOR_DF, RESPONSE, log_density_jax

PARAMETERS_FILE = FOLDER / 'fitted_parameters.json'
STEPS = 4000
VI_SETTINGS = dict(
    steps=STEPS, batch_size=2048, holdout_size=20000,
    average_last=500, learning_rates=(.01, .002), change_after=3000,
)
# Initialize preconditioning_matrix=sqrt(D)*I, then fit the full Cholesky factor and center.
METHODS = {
    'stereographic': {'seed': 26092801, 'radius': np.sqrt(D), 'chunk_size': 250},
}


if __name__ == '__main__':
    run_vi(log_density_jax, D, METHODS, output_json=PARAMETERS_FILE,
           target_info={'response': RESPONSE, 'error_df': NU,
                        'scale_coordinate': 'v=sinh(log_sigma)',
                        'log_sigma_prior': 'flat',
                        'beta_prior': 'product_standard_student_t',
                        'beta_prior_df': PRIOR_DF},
           **VI_SETTINGS)
