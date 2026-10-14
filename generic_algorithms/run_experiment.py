"""Run repeated fixed-parameter couplings for any target log density."""

import csv
from pathlib import Path
import time

import numpy as np

from .algorithms import coupled_sampler


def run_experiment(log_pdf, initial_pairs, method_settings, n_steps, *, seed=42,
                   stop_at_meeting=False, output_csv=None, progress_every=25):
    """Run each method on every initial pair, sequentially.

    ``initial_pairs`` has shape (repetitions, 2, dimension). Each entry in
    ``method_settings`` maps a method name to ``{'h': step_size,
    'parameters': fixed_map_parameters}``. ``latitude`` is optional for SCS,
    and ``seed_offset`` can preserve an application's previous random streams.
    Returns one result dictionary per method and repetition. Unmet meeting
    times are recorded as -1 in the returned rows and optional CSV.
    """
    pairs = np.asarray(initial_pairs, dtype=float)
    if pairs.ndim != 3 or pairs.shape[0] == 0 or pairs.shape[1] != 2 or pairs.shape[2] == 0:
        raise ValueError('initial_pairs must have shape (repetitions, 2, dimension).')
    if not np.isfinite(pairs).all():
        raise ValueError('Initial states must be finite.')
    if not method_settings:
        raise ValueError('Supply at least one method and its settings.')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')

    rows = []
    start = time.perf_counter()
    for rep, (x, y) in enumerate(pairs):
        for method_index, (method, settings) in enumerate(method_settings.items()):
            if 'seed_offset' in settings:
                rng = np.random.default_rng(seed + rep + settings['seed_offset'])
            else:
                rng = np.random.default_rng(np.random.SeedSequence([seed, method_index, rep]))
            result = coupled_sampler(
                x, y, log_pdf, settings['h'], n_steps, rng, method=method,
                parameters=settings.get('parameters'),
                latitude=settings.get('latitude', 1.1),
                stop_at_meeting=stop_at_meeting,
            )
            tau = result['meeting_time']
            rows.append(dict(
                method=method, replicate=rep, met=bool(np.isfinite(tau)),
                meeting_time=int(tau) if np.isfinite(tau) else -1,
                iterations=result['iterations'],
                acceptance_rate_chain1=result['acceptance_rate_chain1'],
                acceptance_rate_chain2=result['acceptance_rate_chain2'],
            ))
        if progress_every and ((rep + 1) % progress_every == 0 or rep + 1 == len(pairs)):
            print(f'Finished {rep + 1}/{len(pairs)} repetitions in '
                  f'{time.perf_counter() - start:.1f}s', flush=True)

    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows
