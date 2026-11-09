"""Target-independent maps. Pass xp=jax.numpy to use the same formulas in VI."""
import numpy as np

METHODS = ('euclidean', 'stereographic', 'scs', 'dcs')


def default_parameters(dimension, method, radius=1.0, *, beta=2.0):
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
        validate_dcs_beta(beta)
        parameters['beta'] = float(beta)
    return parameters


def _validate_location(mu, dimension):
    mu = np.asarray(mu)
    if mu.shape != (dimension,) or not np.isfinite(mu).all():
        raise ValueError('mu must be a finite vector of the target dimension.')


def _validate_matrix(scale, dimension):
    scale = np.asarray(scale)
    if (scale.shape != (dimension, dimension) or not np.isfinite(scale).all()
            or not np.array_equal(scale, np.tril(scale)) or np.any(np.diag(scale) <= 0)):
        raise ValueError('preconditioning_matrix must be a finite lower-triangular matrix '
                         'with positive diagonal.')


def _validate_radius(radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive.')


def validate_dcs_beta(beta):
    """The radial DCS exponent is a fixed positive scalar, independent of dimension."""
    value = np.asarray(beta)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError('DCS beta must be a finite positive scalar.')


def _validate_offset(offset, dimension, limit, name):
    offset = np.asarray(offset)
    if offset.shape != (dimension,) or not np.isfinite(offset).all() or np.linalg.norm(offset) >= limit:
        raise ValueError(f'{name} must have dimension {dimension} and norm below {limit}.')


def _validate_sphere_scale(parameters, dimension):
    """Check a sphere map's matrix scale or its older saved scalar scale."""
    if 'preconditioning_matrix' in parameters:
        if 'radius' in parameters:
            raise ValueError('Supply preconditioning_matrix or legacy radius, not both.')
        _validate_matrix(parameters['preconditioning_matrix'], dimension)
    else:
        _validate_radius(parameters.get('radius', np.nan))


def validate_stereographic_parameters(parameters, dimension):
    """Check the location and scale of one stereographic map."""
    _validate_location(parameters['mu'], dimension)
    _validate_sphere_scale(parameters, dimension)


def validate_scs_parameters(parameters, dimension, latitude=1.1):
    """Check an SCS map, its cap latitude, and its observer."""
    _validate_location(parameters['mu'], dimension)
    _validate_sphere_scale(parameters, dimension)
    if not 0 < latitude < 2:
        raise ValueError('SCS latitude must lie strictly between 0 and 2.')
    limit = np.sqrt(latitude * (2 - latitude))
    _validate_offset(parameters['horizontal_observer'], dimension, limit, 'horizontal_observer')


def validate_dcs_parameters(parameters, dimension):
    """Check a scalar-radius Mobius DCS map."""
    _validate_location(parameters['mu'], dimension)
    validate_dcs_beta(parameters.get('beta', 2.0))
    if 'preconditioning_matrix' in parameters:
        raise ValueError('Mobius DCS uses radius and delta; use the matrix validator for a matrix map.')
    _validate_radius(parameters.get('radius', np.nan))
    _validate_offset(parameters['delta'], dimension, 1., 'delta')


def validate_dcs_matrix_parameters(parameters, dimension):
    """Check a DCS matrix radial map, which has no scalar radius or Mobius offset."""
    _validate_location(parameters['mu'], dimension)
    validate_dcs_beta(parameters.get('beta', 2.0))
    if 'radius' in parameters or 'delta' in parameters:
        raise ValueError('Matrix DCS uses preconditioning_matrix without radius or delta.')
    _validate_matrix(parameters['preconditioning_matrix'], dimension)


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


def dcs_mobius_map(s, parameters, xp=np):
    """Mobius ball automorphism followed by the radial map with fixed beta."""
    mu = parameters['mu']
    d = mu.shape[-1]
    norm2 = xp.sum(s * s, axis=-1)
    radius, delta = parameters['radius'], parameters['delta']
    delta2 = xp.sum(delta * delta)
    denominator = 1 + 2 * xp.sum(s * delta, axis=-1) + delta2 * norm2
    gap = (1 - delta2) * (1 - norm2) / denominator
    beta = parameters.get('beta', 2.0)
    transformed = mobius(s, delta, xp)
    # Use 1-||mobius(s)||^2 directly: recomputing its norm can round to one.
    safe_gap = xp.where(gap < 1, gap, 0.5)
    log_gap = xp.log(-xp.expm1((beta / 2) * xp.log1p(-safe_gap)))
    log_gap = xp.where(gap < 1, log_gap, 0.0)
    x = mu + radius * transformed * xp.exp(-log_gap[..., None] / beta)
    log_jacobian = (d * xp.log(radius) + d * xp.log1p(-delta2)
                    - d * xp.log(denominator) - (1 + d / beta) * log_gap)
    return x, log_jacobian


def dcs_matrix_map(s, parameters, xp=np):
    """x = mu + L s / (1 - ||s||**beta)**(1/beta), including log Jacobian."""
    mu = parameters['mu']
    d = mu.shape[-1]
    norm2 = xp.sum(s * s, axis=-1)
    scale = parameters['preconditioning_matrix']
    beta = parameters.get('beta', 2.0)
    log_gap = dcs_log_radial_gap(norm2, beta, xp)
    radial = s * xp.exp(-log_gap[..., None] / beta)
    x = mu + xp.matmul(radial, xp.swapaxes(scale, -1, -2))
    log_jacobian = (xp.sum(xp.log(xp.diag(scale)))
                    - (1 + d / beta) * log_gap)
    return x, log_jacobian


def dcs_mobius_inverse(x, parameters, xp=np):
    u = (x - parameters['mu']) / parameters['radius']
    return mobius(dcs_radial_inverse(u, parameters.get('beta', 2.0), xp),
                  -parameters['delta'], xp)


def dcs_matrix_inverse(x, parameters, xp=np):
    u = xp.linalg.solve(parameters['preconditioning_matrix'],
                        (x - parameters['mu'])[..., None])[..., 0]
    return dcs_radial_inverse(u, parameters.get('beta', 2.0), xp)


def dcs_log_radial_gap(norm2, beta, xp=np):
    """Stable log(1 - r**beta) for squared radii strictly below one."""
    # Keep both branches finite at the origin, including under JAX autodiff.
    safe_norm2 = xp.where(norm2 > 0, norm2, 0.5)
    log_gap = xp.log(-xp.expm1((beta / 2) * xp.log(safe_norm2)))
    return xp.where(norm2 > 0, log_gap, 0.0)


def dcs_radial_inverse(u, beta=2.0, xp=np):
    """R^d -> B(1), before undoing an optional Mobius automorphism."""
    # Scaling before taking a norm avoids overflow for large Euclidean inputs.
    largest = xp.max(xp.abs(u), axis=-1, keepdims=True)
    safe_largest = xp.where(largest > 0, largest, 1.0)
    scaled_norm2 = xp.sum((u / safe_largest)**2, axis=-1, keepdims=True)
    safe_norm2 = xp.where(scaled_norm2 > 0, scaled_norm2, 1.0)
    log_radius = xp.log(safe_largest) + 0.5 * xp.log(safe_norm2)
    log_denominator = xp.logaddexp(0.0, beta * log_radius) / beta
    log_denominator = xp.where(largest > 0, log_denominator, 0.0)
    return u * xp.exp(-log_denominator)


def dcs_map(s, parameters, xp=np):
    """Compatibility helper for maps loaded in either saved representation."""
    if 'preconditioning_matrix' in parameters:
        return dcs_matrix_map(s, parameters, xp)
    return dcs_mobius_map(s, parameters, xp)


def dcs_inverse(x, parameters, xp=np):
    """Compatibility helper; samplers call the map-specific inverse directly."""
    if 'preconditioning_matrix' in parameters:
        return dcs_matrix_inverse(x, parameters, xp)
    return dcs_mobius_inverse(x, parameters, xp)


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

