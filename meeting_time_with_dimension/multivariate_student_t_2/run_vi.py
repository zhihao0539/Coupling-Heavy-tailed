"""Fit all maps with mu=0, zero SCS horizontal observers, and beta=2 Mobius DCS."""
import argparse
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

FOLDER = Path(__file__).resolve().parent
REPOSITORY = FOLDER.parents[1]
sys.path.insert(0, str(REPOSITORY))

from target import DIMENSIONS, make_log_density, target_info
from generic_algorithms import run_stereographic_vi, run_scs_vi, run_dcs_vi, load_parameters

RESULTS = FOLDER / "results"
SEED = 20261027
DCS_BETAS = {d: 2.0 for d in DIMENSIONS}  # Fixed during VI.
VI_SETTINGS = dict(
    steps=20000, batch_size=2048, holdout_size=20000,
    average_last=500, learning_rates=(0.01, 0.002),
    change_after=10000, chunk_size=250,
)


def fit_stereographic(d, path):
    run_stereographic_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        preconditioning_matrix=np.sqrt(d) * np.eye(d), tune="preconditioning_matrix",
        seed=SEED + 1000*d, output_json=path, target_info=target_info(d), **VI_SETTINGS)


def fit_scs11(d, path):
    run_scs_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        preconditioning_matrix=(np.sqrt(d)/11) * np.eye(d),
        horizontal_observer=np.zeros(d), latitude=1.1, label="SCS11",
        tune="preconditioning_matrix", seed=SEED + 1000*d + 1,
        output_json=path, target_info=target_info(d), **VI_SETTINGS)


def fit_scs19(d, path):
    run_scs_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        preconditioning_matrix=(9*np.sqrt(d)/19) * np.eye(d),
        horizontal_observer=np.zeros(d), latitude=1.9, label="SCS19",
        tune="preconditioning_matrix", seed=SEED + 1000*d + 2,
        output_json=path, target_info=target_info(d), **VI_SETTINGS)


def fit_dcs(d, path):
    run_dcs_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d), radius=1.0,
        delta=np.zeros(d), beta=DCS_BETAS[d], tune=("radius", "delta"),
        seed=SEED + 1000*d + 3, output_json=path, target_info=target_info(d), **VI_SETTINGS)


def fit_dimension(d, *, scs_dcs_only=False):
    folder = RESULTS / f"d_{d:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fitted_parameters.json"
    with TemporaryDirectory(dir=folder) as tmp:
        new_fit = Path(tmp) / path.name
        if scs_dcs_only:
            stereo = load_parameters(path, d, ["stereographic"])["stereographic"]
            if not np.array_equal(stereo["mu"], np.zeros(d)):
                raise ValueError(
                    "The saved stereographic fit has nonzero mu. "
                    "Run python run_vi.py to refit all maps with mu=0.")
            shutil.copyfile(path, new_fit)
        else:
            fit_stereographic(d, new_fit)
        fit_scs11(d, new_fit)
        fit_scs19(d, new_fit)
        fit_dcs(d, new_fit)
        new_fit.replace(path)
    print(f"d={d}: saved VI maps to {path}", flush=True)


def main(scs_dcs_only=False):
    for d in DIMENSIONS:
        fit_dimension(d, scs_dcs_only=scs_dcs_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scs-dcs-only", action="store_true",
                        help="Refit SCS/DCS, preserving stereographic only if its saved mu is exactly zero.")
    main(scs_dcs_only=parser.parse_args().scs_dcs_only)
