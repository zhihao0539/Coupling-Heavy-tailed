"""Step 1: fit matrix stereographic/SCS maps and scalar-radius Mobius DCS."""
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import run_stereographic_vi, run_scs_vi, run_dcs_vi
from skew_t.target import D, NU, log_density_jax

PARAMETERS_FILE = FOLDER / 'fitted_parameters.json'
LATITUDE = 1.1
STEPS = 4000
VI_SETTINGS = dict(
    steps=STEPS, batch_size=2048, holdout_size=20000,
    average_last=500, learning_rates=(.01, .002), change_after=3000,
)
# Matrix methods fit the center and all lower-triangular matrix entries.
# SCS also fits its horizontal observer; DCS fits its center, radius and delta.
# LATITUDE and the DCS exponent beta=2 stay fixed.
def main():
    run_stereographic_vi(
        log_density_jax, D, mu=np.zeros(D), preconditioning_matrix=np.sqrt(D) * np.eye(D),
        seed=26092801, chunk_size=250, output_json=PARAMETERS_FILE,
        target_info={'df': NU}, **VI_SETTINGS)
    run_scs_vi(
        log_density_jax, D, mu=np.zeros(D),
        preconditioning_matrix=(np.sqrt(D) / 11) * np.eye(D),
        horizontal_observer=np.zeros(D), latitude=LATITUDE,
        seed=26093001, chunk_size=250, output_json=PARAMETERS_FILE,
        target_info={'df': NU}, **VI_SETTINGS)
    run_dcs_vi(
        log_density_jax, D, mu=np.zeros(D), radius=np.sqrt(NU), delta=np.zeros(D),
        seed=26092901, chunk_size=STEPS, output_json=PARAMETERS_FILE,
        target_info={'df': NU}, **VI_SETTINGS)


if __name__ == '__main__':
    main()
