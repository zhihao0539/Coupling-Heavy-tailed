"""Run one explicit VI fit and save it; loading fits keeps JAX optional."""
import json
from pathlib import Path

import numpy as np

from .transformations import (
    validate_stereographic_parameters, validate_scs_parameters,
    validate_dcs_parameters, validate_dcs_matrix_parameters,
    validate_dcs_beta,
)


def run_stereographic_vi(log_pdf, dimension, *, mu, preconditioning_matrix, tune=None,
        output_json=None, label='stereographic', target_info=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit and optionally save one stereographic map using its uniform sphere reference."""
    from .vi import fit_stereographic_vi

    print(f'Fitting the {label.upper()} map by VI', flush=True)
    result = fit_stereographic_vi(log_pdf, dimension,
        mu=mu, preconditioning_matrix=preconditioning_matrix,
        tune=tune, seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    initial = dict(mu=mu, preconditioning_matrix=preconditioning_matrix)
    return _finish_fit(result, initial, label, output_json, target_info)


def run_scs_vi(log_pdf, dimension, *, mu, preconditioning_matrix,
        horizontal_observer, latitude=1.1, tune=None,
        output_json=None, label='scs', target_info=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit and optionally save one SCS map using its uniform bright-cap reference."""
    from .vi import fit_scs_vi

    print(f'Fitting the {label.upper()} map by VI', flush=True)
    result = fit_scs_vi(log_pdf, dimension,
        mu=mu, preconditioning_matrix=preconditioning_matrix,
        horizontal_observer=horizontal_observer, latitude=latitude,
        tune=tune, seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    initial = dict(mu=mu, preconditioning_matrix=preconditioning_matrix,
                   horizontal_observer=horizontal_observer)
    return _finish_fit(result, initial, label, output_json, target_info)


def run_dcs_vi(log_pdf, dimension, *, mu, radius, delta, beta=2.0, tune=None,
        output_json=None, label='dcs', target_info=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit and optionally save one Mobius DCS map using uniform unit-ball points."""
    from .vi import fit_dcs_vi

    print(f'Fitting the {label.upper()} map by VI', flush=True)
    result = fit_dcs_vi(log_pdf, dimension,
        mu=mu, radius=radius, delta=delta, beta=beta,
        tune=tune, seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    initial = dict(mu=mu, radius=radius, delta=delta, beta=beta)
    return _finish_fit(result, initial, label, output_json, target_info)


def run_dcs_matrix_vi(log_pdf, dimension, *, mu, preconditioning_matrix, beta=2.0, tune=None,
        output_json=None, label='dcs', target_info=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit and optionally save one matrix DCS map using uniform unit-ball points."""
    from .vi import fit_dcs_matrix_vi

    print(f'Fitting the {label.upper()} map by VI', flush=True)
    result = fit_dcs_matrix_vi(log_pdf, dimension,
        mu=mu, preconditioning_matrix=preconditioning_matrix, beta=beta,
        tune=tune, seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    initial = dict(mu=mu, preconditioning_matrix=preconditioning_matrix, beta=beta)
    return _finish_fit(result, initial, label, output_json, target_info)


def _finish_fit(result, initial, label, output_json, target_info):
    """Report and save a completed fit."""
    parameters, info, history = result
    _add_initial_info(info, initial, target_info)
    _print_fit_summary(parameters, info)
    if output_json is not None:
        save_parameters({label: parameters}, {label: info}, output_json)
    return parameters, info, history


def _geometric_scale(parameters):
    return float(np.exp(np.mean(np.log(np.diag(parameters['preconditioning_matrix'])))))


def _add_initial_info(info, initial_parameters, target_info):
    """Record the starting scale and the application's optional description."""
    if 'preconditioning_matrix' in initial_parameters:
        info['initial_scale_geometric_mean'] = _geometric_scale(initial_parameters)
    else:
        info['initial_radius'] = float(initial_parameters['radius'])
    if target_info is not None:
        info['target_info'] = dict(target_info)


def _print_fit_summary(parameters, info):
    if 'preconditioning_matrix' in parameters:
        scale_summary = f'geometric scale {_geometric_scale(parameters):.6f}'
    else:
        scale_summary = f"radius {parameters['radius']:.6f}"
    if 'beta' in info['tuned_parameters']:
        scale_summary += f", beta {parameters['beta']:.6f}"
    print(f"  {scale_summary}, chosen {info['chosen_candidate']}, "
          f"holdout losses {info['candidate_holdout_losses']}", flush=True)


def save_parameters(parameters, info, path):
    """Save fits and VI details, preserving existing methods that were not refitted."""
    path = Path(path)
    record = json.loads(path.read_text()) if path.exists() else {}
    vi_info = record.pop('vi_info', {})
    for method, values in parameters.items():
        record[method] = {name: float(value) if name in ('radius', 'beta') else np.asarray(value).tolist()
                          for name, value in values.items()}
        vi_info[method] = info[method]
    record['vi_info'] = vi_info
    contents = json.dumps(record, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    print(f"Saved the {', '.join(m.upper() for m in parameters)} fit to {path}", flush=True)


def load_parameters(path, dimension, methods, *, latitude=1.1, legacy_scs_latitude=None,
                    beta=None):
    """Load maps without JAX; optionally require a particular fitted DCS beta.

    For an older SCS file without latitude metadata, explicitly supply its known
    latitude as legacy_scs_latitude. Older DCS maps without beta use beta=2.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; run the application\'s run_vi.py first.')
    record = json.loads(path.read_text())
    parameters = {}
    for name in methods:
        if name not in record:
            raise ValueError(f'{path.name} has no {name.upper()} fit; run VI for this method.')
        info = record.get('vi_info', {}).get(name, {})
        method = info.get('method', name)
        if method == 'scs':
            _check_scs_latitude(info, latitude, legacy_scs_latitude)
        parameters[name] = {
            field: float(value) if field in ('radius', 'beta') else np.asarray(value, dtype=float)
            for field, value in record[name].items()}
        values = parameters[name]
        if method == 'dcs':
            values['beta'] = _saved_dcs_beta(values, info, beta)
        if method == 'stereographic':
            validate_stereographic_parameters(values, dimension)
        elif method == 'scs':
            validate_scs_parameters(values, dimension, latitude)
        elif method == 'dcs' and 'preconditioning_matrix' in values:
            validate_dcs_matrix_parameters(values, dimension)
        elif method == 'dcs':
            validate_dcs_parameters(values, dimension)
        elif method != 'euclidean':
            raise ValueError(f'Unknown method: {method}')
    return parameters


def _saved_dcs_beta(parameters, info, requested_beta):
    """Retain the fitted exponent and detect inconsistent or mismatched metadata."""
    saved = parameters.get('beta', info.get('beta', 2.0))
    validate_dcs_beta(saved)
    if 'beta' in info:
        validate_dcs_beta(info['beta'])
        if not np.isclose(saved, info['beta'], rtol=0, atol=1e-12):
            raise ValueError('The saved DCS parameters and VI metadata have different beta values.')
    if requested_beta is not None:
        validate_dcs_beta(requested_beta)
        if not np.isclose(saved, requested_beta, rtol=0, atol=1e-12):
            raise ValueError(f'The saved DCS map uses beta={saved}, but beta={requested_beta} '
                             'was requested; refit DCS or load the matching map.')
    return float(saved)


def _check_scs_latitude(info, latitude, legacy_scs_latitude):
    """Require a known saved latitude matching the sampling latitude."""
    saved_latitude = info.get('latitude', legacy_scs_latitude)
    if saved_latitude is None:
        raise ValueError('Saved SCS latitude is unknown. Supply legacy_scs_latitude '
                         'only if the original fit\'s latitude is known, or refit it.')
    if not np.isclose(saved_latitude, latitude, rtol=0, atol=1e-12):
        raise ValueError(f'The saved SCS map was fitted with latitude {saved_latitude}, '
                         f'but the requested latitude is {latitude}; refit the map.')
