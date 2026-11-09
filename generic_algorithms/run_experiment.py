"""Explicit experiment functions, one for each coupling and map family."""
import csv
from pathlib import Path

import numpy as np

from .algorithms import (
    coupled_euclidean, coupled_stereographic, coupled_scs,
    coupled_dcs_ball, coupled_dcs_random_walk,
    coupled_dcs_matrix_ball, coupled_dcs_matrix_random_walk,
)


def run_euclidean_experiment(log_pdf, initial_pairs, h, n_steps, *,
        seed=42, seed_index=0, seed_offset=None, label='euclidean',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """Euclidean Gaussian random walk; h is the coordinate standard deviation."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_euclidean(x, y, log_pdf, h, n_steps, rng,
            stop_at_meeting=stop_at_meeting)
        rows.append(_result_row(label, rep, result))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def run_stereographic_experiment(log_pdf, initial_pairs, h, n_steps, *,
        mu, preconditioning_matrix,
        seed=42, seed_index=1, seed_offset=None, label='stereographic',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """Stereographic coupling with location mu and lower-triangular scale matrix."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_stereographic(x, y, log_pdf, h, n_steps, rng,
            mu=mu, preconditioning_matrix=preconditioning_matrix,
            stop_at_meeting=stop_at_meeting)
        rows.append(_result_row(label, rep, result))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def run_scs_experiment(log_pdf, initial_pairs, h, n_steps, *,
        mu, preconditioning_matrix, horizontal_observer, latitude=1.1,
        seed=42, seed_index=2, seed_offset=None, label='scs',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """SCS coupling with an explicit cap latitude and horizontal observer."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_scs(x, y, log_pdf, h, n_steps, rng,
            mu=mu, preconditioning_matrix=preconditioning_matrix,
            horizontal_observer=horizontal_observer, latitude=latitude,
            stop_at_meeting=stop_at_meeting)
        rows.append(_result_row(label, rep, result))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def run_dcs_ball_experiment(log_pdf, initial_pairs, h, n_steps, *,
        mu, radius, delta, beta=2.0,
        seed=42, seed_index=3, seed_offset=None, label='dcs',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """DCS ball walk with scalar radius and Mobius offset delta; h = gamma/R."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_dcs_ball(x, y, log_pdf, h, n_steps, rng,
            mu=mu, radius=radius, delta=delta, beta=beta, stop_at_meeting=stop_at_meeting)
        rows.append(dict(_result_row(label, rep, result), beta=float(beta)))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def run_dcs_random_walk_experiment(log_pdf, initial_pairs, h, n_steps, *,
        mu, radius, delta, beta=2.0,
        seed=42, seed_index=4, seed_offset=None, label='dcs_rw',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """DCS Gaussian walk with scalar radius and Mobius offset; h = sigma/R."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_dcs_random_walk(x, y, log_pdf, h, n_steps, rng,
            mu=mu, radius=radius, delta=delta, beta=beta, stop_at_meeting=stop_at_meeting)
        rows.append(dict(_result_row(label, rep, result), beta=float(beta)))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def run_dcs_matrix_ball_experiment(log_pdf, initial_pairs, h, n_steps, *,
        mu, preconditioning_matrix, beta=2.0,
        seed=42, seed_index=3, seed_offset=None, label='dcs',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """DCS ball walk with a matrix radial map; h is the native ball-step radius."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_dcs_matrix_ball(x, y, log_pdf, h, n_steps, rng,
            mu=mu, preconditioning_matrix=preconditioning_matrix, beta=beta,
            stop_at_meeting=stop_at_meeting)
        rows.append(dict(_result_row(label, rep, result), beta=float(beta)))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def run_dcs_matrix_random_walk_experiment(log_pdf, initial_pairs, h, n_steps, *,
        mu, preconditioning_matrix, beta=2.0,
        seed=42, seed_index=4, seed_offset=None, label='dcs_rw',
        stop_at_meeting=False, output_csv=None, progress_every=25):
    """DCS Gaussian walk with a matrix radial map; h is the native standard deviation."""
    pairs = _validate_experiment(initial_pairs, seed, seed_index)
    rows = []
    for rep, (x, y) in enumerate(pairs):
        rng = _replicate_rng(seed, seed_index, seed_offset, rep)
        result = coupled_dcs_matrix_random_walk(x, y, log_pdf, h, n_steps, rng,
            mu=mu, preconditioning_matrix=preconditioning_matrix, beta=beta,
            stop_at_meeting=stop_at_meeting)
        rows.append(dict(_result_row(label, rep, result), beta=float(beta)))
        _print_progress(label, rep, len(pairs), progress_every)
    return _finish_experiment(rows, output_csv)


def _print_progress(label, rep, count, progress_every):
    if progress_every and ((rep + 1) % progress_every == 0 or rep + 1 == count):
        print(f'{label}: finished {rep + 1}/{count} repetitions', flush=True)


def _finish_experiment(rows, output_csv):
    if output_csv is not None:
        save_results(rows, output_csv)
    return rows


def _validate_experiment(initial_pairs, seed, seed_index):
    pairs = np.asarray(initial_pairs, dtype=float)
    if pairs.ndim != 3 or pairs.shape[0] == 0 or pairs.shape[1] != 2 or pairs.shape[2] == 0:
        raise ValueError('initial_pairs must have shape (repetitions, 2, dimension).')
    if not np.isfinite(pairs).all():
        raise ValueError('Initial states must be finite.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if not isinstance(seed_index, (int, np.integer)) or seed_index < 0:
        raise ValueError('seed_index must be a nonnegative integer.')
    return pairs


def _replicate_rng(seed, seed_index, seed_offset, rep):
    if seed_offset is not None:
        return np.random.default_rng(seed + rep + seed_offset)
    return np.random.default_rng(np.random.SeedSequence([seed, seed_index, rep]))


def _result_row(label, rep, result):
    tau = result['meeting_time']
    return dict(
        method=label, replicate=rep, met=bool(np.isfinite(tau)),
        meeting_time=int(tau) if np.isfinite(tau) else -1,
        iterations=result['iterations'],
        acceptance_rate_chain1=result['acceptance_rate_chain1'],
        acceptance_rate_chain2=result['acceptance_rate_chain2'],
    )


def combine_results(*groups):
    """Interleave method results by repetition, preserving the supplied method order."""
    return sorted((row for group in groups for row in group), key=lambda row: row['replicate'])


def save_results(rows, path):
    """Write one method's results or combined results in the existing CSV format."""
    if not rows:
        raise ValueError('Supply at least one result row.')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
