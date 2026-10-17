"""Fit one or more VI maps, save them, and load them for coupling experiments."""
import json
from pathlib import Path

import numpy as np

from .transformations import default_parameters, validate_parameters


def run_vi(log_pdf, dimension, method_settings, *, output_json=None,
           latitude=1.1, target_info=None, **vi_settings):
    """Call fit_vi for each method, optionally saving the fitted maps to JSON.

    method_settings maps names to settings, e.g. {'scs': {'radius': 1., 'seed': 7}}.
    Each entry overrides the shared fit_vi keywords in vi_settings. radius is a
    shortcut for zero-centered initial_parameters; for SCS and stereographic
    it initializes preconditioning_matrix=radius*I. Supply one or the other.
    Matrix DCS uses an explicit initial_parameters with mu and preconditioning_matrix.
    Returns three dictionaries keyed by method: parameters, info, and history.
    JAX is imported only when this function is called, not when loading fits.
    """
    from .vi import VI_METHODS, fit_vi

    if not method_settings or any(method not in VI_METHODS for method in method_settings):
        raise ValueError(f'Supply one or more VI methods from {VI_METHODS}.')
    parameters, info, history = {}, {}, {}
    for method, settings in method_settings.items():
        options = dict(vi_settings, **settings)
        options.setdefault('latitude', latitude)
        if 'radius' in options and options.get('initial_parameters') is not None:
            raise ValueError('Supply radius or initial_parameters, not both.')
        radius = options.pop('radius', 1.)
        if options.get('initial_parameters') is None:
            options['initial_parameters'] = default_parameters(dimension, method, radius)
        print(f'Fitting the {method.upper()} map by VI', flush=True)
        parameters[method], info[method], history[method] = fit_vi(
            log_pdf, dimension, method, **options)
        if 'preconditioning_matrix' in options['initial_parameters']:
            scale = options['initial_parameters']['preconditioning_matrix']
            initial_scale = float(np.exp(np.mean(np.log(np.diag(scale)))))
            info[method]['initial_scale_geometric_mean'] = initial_scale
        else:
            info[method]['initial_radius'] = float(options['initial_parameters']['radius'])
        if target_info is not None:
            info[method]['target_info'] = dict(target_info)
        if 'preconditioning_matrix' in parameters[method]:
            scale = parameters[method]['preconditioning_matrix']
            scale_summary = f"geometric scale {np.exp(np.mean(np.log(np.diag(scale)))):.6f}"
        else:
            scale_summary = f"radius {parameters[method]['radius']:.6f}"
        print(f"  {scale_summary}, "
              f"chosen {info[method]['chosen_candidate']}, "
              f"holdout losses {info[method]['candidate_holdout_losses']}", flush=True)
    if output_json is not None:
        save_parameters(parameters, info, output_json)
    return parameters, info, history


def save_parameters(parameters, info, path):
    """Save fits and VI details, preserving existing methods that were not refitted."""
    path = Path(path)
    record = json.loads(path.read_text()) if path.exists() else {}
    vi_info = record.pop('vi_info', {})
    for method, values in parameters.items():
        record[method] = {name: float(value) if name == 'radius' else np.asarray(value).tolist()
                          for name, value in values.items()}
        vi_info[method] = info[method]
    record['vi_info'] = vi_info
    contents = json.dumps(record, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    print(f"Saved the {', '.join(m.upper() for m in parameters)} fit to {path}", flush=True)


def load_parameters(path, dimension, methods, *, latitude=1.1, legacy_scs_latitude=None):
    """Load requested maps without JAX; check dimension, support, and SCS latitude.

    For an older SCS file without latitude metadata, explicitly supply its known
    latitude as legacy_scs_latitude. New files record latitude under vi_info.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; run the application\'s run_vi.py first.')
    record = json.loads(path.read_text())
    parameters = {}
    for method in methods:
        if method not in record:
            raise ValueError(f'{path.name} has no {method.upper()} fit; run VI for this method.')
        info = record.get('vi_info', {}).get(method, {})
        if method == 'scs':
            saved_latitude = info.get('latitude', legacy_scs_latitude)
            if saved_latitude is None:
                raise ValueError('Saved SCS latitude is unknown. Supply legacy_scs_latitude '
                                 'only if the original fit\'s latitude is known, or refit it.')
            if not np.isclose(saved_latitude, latitude, rtol=0, atol=1e-12):
                raise ValueError(f'The saved SCS map was fitted with latitude {saved_latitude}, '
                                 f'but the requested latitude is {latitude}; refit the map.')
        parameters[method] = {
            name: float(value) if name == 'radius' else np.asarray(value, dtype=float)
            for name, value in record[method].items()}
        validate_parameters(parameters[method], dimension, method, latitude)
    return parameters
