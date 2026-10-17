"""Target-independent maps. Pass xp=jax.numpy to use the same formulas in VI."""
import numpy as np

METHODS = ('euclidean', 'stereographic', 'scs', 'dcs')


def default_parameters(dimension, method, radius=1.0):
    if method not in METHODS or dimension < 1:
        raise ValueError('Choose a supported method and a positive dimension.')
    parameters = {'mu': np.zeros(dimension)}
    if method in ('stereographic', 'scs'):
        # L L.T is Sigma; radius remains the isotropic starting scale.
        parameters['preconditioning_matrix'] = float(radius) * np.eye(dimension)
        if method == 'scs':
            parameters['horizontal_observer'] = np.zeros(dimension)
    else:
        parameters['radius'] = float(radius)
    if method == 'dcs':
        parameters['delta'] = np.zeros(dimension)
    return parameters


def validate_parameters(parameters, dimension, method, latitude=1.1):
    """Validate once, outside the sampling/optimization loops."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if method == 'euclidean':
        return
    mu = np.asarray(parameters['mu'])
    if mu.shape != (dimension,) or not np.isfinite(mu).all():
        raise ValueError('mu must be a finite vector of the target dimension.')
    if method in ('stereographic', 'scs', 'dcs') and 'preconditioning_matrix' in parameters:
        if 'radius' in parameters:
            raise ValueError(f'{method} accepts preconditioning_matrix or legacy radius, not both.')
        if method == 'dcs' and 'delta' in parameters:
            raise ValueError('Matrix DCS uses the paper\'s radial map without a Mobius offset.')
        scale = np.asarray(parameters['preconditioning_matrix'])
        if (scale.shape != (dimension, dimension) or not np.isfinite(scale).all()
                or not np.array_equal(scale, np.tril(scale))
                or np.any(np.diag(scale) <= 0)):
            raise ValueError(f'{method} preconditioning_matrix must be a finite lower-triangular matrix '
                             'with positive diagonal.')
    elif 'radius' not in parameters or not np.isfinite(parameters['radius']) or parameters['radius'] <= 0:
        raise ValueError('radius must be finite and positive.')
    if method == 'scs' and not 0 < latitude < 2:
        raise ValueError('SCS latitude must lie strictly between 0 and 2.')
    if method == 'scs' or (method == 'dcs' and 'preconditioning_matrix' not in parameters):
        name = 'horizontal_observer' if method == 'scs' else 'delta'
        offset = np.asarray(parameters[name])
        limit = np.sqrt(latitude * (2 - latitude)) if method == 'scs' else 1.
        if offset.shape != (dimension,) or not np.isfinite(offset).all() or np.linalg.norm(offset) >= limit:
            raise ValueError(f'{name} must have dimension {dimension} and norm below {limit}.')


def stereographic_map(z, parameters, xp=np):
    """Sphere -> R^d, returning the log Jacobian relative to sphere area."""
    mu = parameters['mu']
    d = mu.shape[-1]
    gap = 1 - z[..., -1]
    x_hat = z[..., :-1] / gap[..., None]
    if 'preconditioning_matrix' in parameters:
        scale = parameters['preconditioning_matrix']
        x = mu + xp.matmul(x_hat, xp.swapaxes(scale, -1, -2))
        log_scale_volume = xp.sum(xp.log(xp.diag(scale)))
    else:                          # Previously saved scalar maps still load.
        radius = parameters['radius']
        x = mu + radius * x_hat
        log_scale_volume = d * xp.log(radius)
    return x, log_scale_volume - d * xp.log(gap)


def stereographic_inverse(x, parameters, xp=np):
    centered = x - parameters['mu']
    if 'preconditioning_matrix' in parameters:
        u = xp.linalg.solve(parameters['preconditioning_matrix'], centered[..., None])[..., 0]
    else:
        u = centered / parameters['radius']
    norm2 = xp.sum(u * u, axis=-1, keepdims=True)
    return xp.concatenate((2 * u / (1 + norm2), (norm2 - 1) / (1 + norm2)), axis=-1)


def scs_map(z, parameters, latitude=1.1, xp=np):
    """General SCP map, storing w-e_(d+1) as the native unit-sphere point.

    The projection observer is o=(horizontal_observer, latitude) on the
    sphere centered at e_(d+1). preconditioning_matrix is a square root of Sigma.
    """
    mu, observer = parameters['mu'], parameters['horizontal_observer']
    d = mu.shape[-1]
    height = z[..., -1]
    gap = latitude - 1 - height
    x_hat = (latitude * z[..., :-1] - (1 + height[..., None]) * observer) / gap[..., None]
    if 'preconditioning_matrix' in parameters:
        scale = parameters['preconditioning_matrix']
        x = mu + xp.matmul(x_hat, xp.swapaxes(scale, -1, -2))
        log_scale_volume = xp.sum(xp.log(xp.diag(scale)))
    else:                          # Previously saved scalar SCS maps still load.
        radius = parameters['radius']
        x = mu + radius * x_hat
        log_scale_volume = d * xp.log(radius)
    cosine = 1 - (latitude - 1) * height - xp.sum(z[..., :-1] * observer, axis=-1)
    log_jacobian = (log_scale_volume + d * xp.log(latitude)
                    + xp.log(cosine) - (d + 1) * xp.log(gap))
    return x, log_jacobian


def scs_inverse(x, parameters, latitude=1.1, xp=np):
    mu, observer = parameters['mu'], parameters['horizontal_observer']
    b = latitude - 1
    # Solve L x_hat = x-mu; below is the positive M root in the supplied inverse.
    if 'preconditioning_matrix' in parameters:
        x_hat = xp.linalg.solve(parameters['preconditioning_matrix'], (x - mu)[..., None])[..., 0]
    else:
        x_hat = (x - mu) / parameters['radius']
    a = x_hat - observer
    aa = xp.sum(a * a, axis=-1) + latitude**2
    half_b = xp.sum(a * observer, axis=-1) - latitude * b
    cc = xp.sum(observer * observer) + b**2 - 1
    root = xp.sqrt(half_b**2 - aa * cc)
    # Both denominators stay nonzero, including the unused branch under JAX.
    m = xp.where(half_b > 0, -cc / (root + xp.maximum(half_b, 0)),
                 (-half_b + root) / aa)
    return xp.concatenate((m[..., None] * a + observer, (b - latitude * m)[..., None]), axis=-1)


def mobius(s, delta, xp=np):
    norm2 = xp.sum(s * s, axis=-1)
    dot = xp.sum(s * delta, axis=-1)
    delta2 = xp.sum(delta * delta)
    denominator = 1 + 2 * dot + delta2 * norm2
    return ((1 - delta2) * s + (1 + 2 * dot + norm2)[..., None] * delta) / denominator[..., None]


def dcs_map(s, parameters, xp=np):
    """Unit ball -> R^d, with the paper's matrix radial map or legacy Mobius map."""
    mu = parameters['mu']
    d = mu.shape[-1]
    norm2 = xp.sum(s * s, axis=-1)
    if 'preconditioning_matrix' in parameters:
        # Appendix B: mu + L RC_2(s), where L absorbs sqrt(nu) Sigma^(1/2).
        scale = parameters['preconditioning_matrix']
        radial = s / xp.sqrt((1 - norm2)[..., None])
        x = mu + xp.matmul(radial, xp.swapaxes(scale, -1, -2))
        log_jacobian = (xp.sum(xp.log(xp.diag(scale)))
                        - (1 + d / 2) * xp.log1p(-norm2))
        return x, log_jacobian
    radius, delta = parameters['radius'], parameters['delta']
    delta2 = xp.sum(delta * delta)
    denominator = 1 + 2 * xp.sum(s * delta, axis=-1) + delta2 * norm2
    gap = (1 - delta2) * (1 - norm2) / denominator
    x = mu + radius * mobius(s, delta, xp) / xp.sqrt(gap[..., None])
    log_jacobian = d * xp.log(radius) + d * xp.log1p(-delta2) - d * xp.log(denominator) - (1 + d / 2) * xp.log(gap)
    return x, log_jacobian


def dcs_inverse(x, parameters, xp=np):
    if 'preconditioning_matrix' in parameters:
        u = xp.linalg.solve(parameters['preconditioning_matrix'],
                            (x - parameters['mu'])[..., None])[..., 0]
        return u / xp.sqrt(1 + xp.sum(u * u, axis=-1, keepdims=True))
    u = (x - parameters['mu']) / parameters['radius']
    return mobius(u / xp.sqrt(1 + xp.sum(u * u, axis=-1, keepdims=True)), -parameters['delta'], xp)


def map_to_euclidean(z, method, parameters, latitude=1.1, xp=np):
    if method == 'stereographic':
        return stereographic_map(z, parameters, xp)
    if method == 'scs':
        return scs_map(z, parameters, latitude, xp)
    if method == 'dcs':
        return dcs_map(z, parameters, xp)
    if method == 'euclidean':
        return z, xp.zeros(z.shape[:-1])
    raise ValueError(f'Unknown method: {method}')


def to_native(x, method, parameters=None, latitude=1.1):
    """Convert an application state to the state used by its proposal."""
    x = np.asarray(x, dtype=float)
    if method == 'stereographic':
        return stereographic_inverse(x, parameters)
    if method == 'scs':
        return scs_inverse(x, parameters, latitude)
    if method == 'dcs':
        return dcs_inverse(x, parameters)
    if method == 'euclidean':
        return x.copy()
    raise ValueError(f'Unknown method: {method}')


def from_native(z, method, parameters=None, latitude=1.1):
    return map_to_euclidean(z, method, parameters, latitude)[0]


def native_log_pdf(z, log_pdf, method, parameters=None, latitude=1.1):
    """Density in proposal coordinates, including the transformation Jacobian."""
    if method == 'scs' and z[-1] >= latitude - 1:
        return -np.inf                         # Reject dark side, without stepping out.
    if method == 'dcs' and z @ z >= 1:
        return -np.inf                         # Reject outside the unit ball.
    if method == 'stereographic' and z[-1] >= 1:
        return -np.inf
    x, log_jacobian = map_to_euclidean(z, method, parameters, latitude)
    value = float(log_pdf(x) + log_jacobian)
    if np.isnan(value) or np.isposinf(value):
        raise FloatingPointError('log_pdf + log Jacobian must be finite or -inf.')
    return value

