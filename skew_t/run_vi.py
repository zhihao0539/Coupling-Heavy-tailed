"""Tune the skew-t SCS and DCS maps using the shared VI functions."""
import json
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import default_parameters
from generic_algorithms.vi import fit_vi
import jax.numpy as jnp
from jax.scipy.special import betainc

D = 100
DF = 2.0
LATITUDE = 1.1
STEPS = 4000
BATCH_SIZE = 2048
HOLDOUT_SIZE = 20000
AVERAGE_LAST = 500
LEARNING_RATES = (0.01, 0.002)
CHANGE_AFTER = 3000
METHODS = {
    'scs': {'seed': 26093001, 'radius': 10 / 11},
    'dcs': {'seed': 26092901, 'radius': np.sqrt(DF)},
}


def log_target(x):
    """Unnormalized AC skew-t density, alpha = (100, -100, 0, ...)."""
    norm2 = jnp.sum(x * x, axis=-1)
    t = 100 * (x[..., 0] - x[..., 1]) * jnp.sqrt((D + DF) / (DF + norm2))
    df = D + DF
    w = jnp.clip(df / (df + t * t), 0.0, 1.0 - jnp.finfo(jnp.float64).eps)
    b = betainc(df / 2, 0.5, w)
    log_cdf = jnp.where(t < 0,
                        jnp.log(0.5) + jnp.log(jnp.maximum(b, jnp.finfo(jnp.float64).tiny)),
                        jnp.log1p(-0.5 * b))
    return -(D + DF) / 2 * jnp.log1p(norm2 / DF) + log_cdf


def main():
    for method, settings in METHODS.items():
        parameters, info, _ = fit_vi(
            log_target, D, method,
            initial_parameters=default_parameters(D, method, radius=settings['radius']),
            latitude=LATITUDE, seed=settings['seed'], steps=STEPS,
            batch_size=BATCH_SIZE, holdout_size=HOLDOUT_SIZE,
            average_last=AVERAGE_LAST, learning_rates=LEARNING_RATES,
            change_after=CHANGE_AFTER, chunk_size=250 if method == 'scs' else STEPS,
        )
        info.update(df=DF, initial_radius=settings['radius'])
        parameters = {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in parameters.items()}
        output = FOLDER / f'{method}_fit.json'
        output.write_text(json.dumps({'parameters': parameters, 'fit': info}, indent=2) + '\n')
        print(f'Saved {output.name}', flush=True)


if __name__ == '__main__':
    main()
