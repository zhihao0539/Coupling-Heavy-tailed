"""Fit the skew-t SCS and DCS maps using the shared VI runner."""
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import run_vi
from skew_t.target import D, NU, log_density_jax

PARAMETERS_FILE = FOLDER / 'fitted_parameters.json'
LATITUDE = 1.1
STEPS = 4000
VI_SETTINGS = dict(
    steps=STEPS, batch_size=2048, holdout_size=20000,
    average_last=500, learning_rates=(.01, .002), change_after=3000,
)
# Initial location and offsets are zero. Remove an entry to refit only one map;
# the shared runner preserves saved maps that are not refitted.
METHODS = {
    'scs': {'seed': 26093001, 'radius': 10 / 11, 'chunk_size': 250},
    'dcs': {'seed': 26092901, 'radius': np.sqrt(NU), 'chunk_size': STEPS},
}


if __name__ == '__main__':
    run_vi(log_density_jax, D, METHODS, output_json=PARAMETERS_FILE,
           latitude=LATITUDE, target_info={'df': NU}, **VI_SETTINGS)
