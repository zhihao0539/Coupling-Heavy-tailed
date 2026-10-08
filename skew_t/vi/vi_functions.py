"""Reverse-KL variational fitting for the skew-t SCS and DCS maps."""
import json
import math
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import betainc

D = 100
DF = 2.0
LATITUDE = 1.1                    # SCS latitude; the DCS map below uses beta = 2.


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


def unpack(theta, method):
    """Convert 201 unconstrained variables into a location, radius and offset."""
    mu = theta[:D]
    radius = jnp.exp(theta[D])
    v = theta[D + 1:]
    norm = jnp.sqrt(jnp.sum(v * v) + 1e-24)
    limit = math.sqrt(1 - (LATITUDE - 1)**2) if method == "scs" else 1.0
    offset = limit * (jnp.tanh(norm) / norm) * v
    return mu, radius, offset


def scs_map(z, theta):
    mu, radius, observer = unpack(theta, "scs")
    height = z[..., -1]
    gap = LATITUDE - 1 - height
    x = mu + radius * (LATITUDE * z[..., :-1] - (1 + height[..., None]) * observer) / gap[..., None]
    cosine = 1 - (LATITUDE - 1) * height - jnp.sum(z[..., :-1] * observer, axis=-1)
    log_jacobian = D * jnp.log(radius * LATITUDE) + jnp.log(cosine) - (D + 1) * jnp.log(gap)
    return x, log_jacobian


def dcs_map(s, theta):
    """Mobius transformation followed by the beta=2 map from the unit ball."""
    mu, radius, delta = unpack(theta, "dcs")
    norm2 = jnp.sum(s * s, axis=-1)
    dot = jnp.sum(s * delta, axis=-1)
    delta2 = jnp.sum(delta * delta)
    denominator = 1 + 2 * dot + delta2 * norm2
    y = ((1 - delta2) * s + (1 + 2 * dot + norm2)[..., None] * delta) / denominator[..., None]
    gap = (1 - delta2) * (1 - norm2) / denominator
    x = mu + radius * y / jnp.sqrt(gap[..., None])
    log_jacobian = D * jnp.log(radius) + D * jnp.log1p(-delta2) - D * jnp.log(denominator) - (1 + D / 2) * jnp.log(gap)
    return x, log_jacobian


def uniform_cap(key, n):
    """IID uniform samples on the fixed bright cap z[-1] < latitude - 1."""
    def condition(state):
        return state[2] < n

    def draw(state):
        key, subkey = jax.random.split(state[0])
        g = jax.random.normal(subkey, (2 * n, D + 1), dtype=jnp.float64)
        z = g / jnp.linalg.norm(g, axis=1, keepdims=True)
        valid = z[:, -1] < LATITUDE - 1
        indices = jnp.nonzero(valid, size=n, fill_value=0)[0]
        return key, z[indices], jnp.sum(valid, dtype=jnp.int32)

    initial = (key, jnp.zeros((n, D + 1)), jnp.int32(0))
    return jax.lax.while_loop(condition, draw, initial)[1]


def uniform_ball(key, n):
    direction_key, radius_key = jax.random.split(key)
    g = jax.random.normal(direction_key, (n, D), dtype=jnp.float64)
    radii = jax.random.uniform(radius_key, (n, 1), dtype=jnp.float64)**(1 / D)
    return g / jnp.linalg.norm(g, axis=-1, keepdims=True) * radii


def loss(theta, samples, method):
    """KL(pushforward reference || target), up to parameter-free constants."""
    x, log_jacobian = scs_map(samples, theta) if method == "scs" else dcs_map(samples, theta)
    return -jnp.mean(log_target(x) + log_jacobian)


def fit(method, seed, steps=4000, batch_size=2048, holdout_size=20000,
        average_last=500, learning_rates=(0.01, 0.002), change_after=3000):
    if method not in ("scs", "dcs"):
        raise ValueError("method must be 'scs' or 'dcs'")
    if min(steps, batch_size, holdout_size, average_last) < 1:
        raise ValueError("Iteration and sample counts must be positive")
    sample = uniform_cap if method == "scs" else uniform_ball
    initial_radius = 10 / 11 if method == "scs" else math.sqrt(DF)
    theta0 = jnp.zeros(2 * D + 1).at[D].set(math.log(initial_radius))
    objective = jax.jit(lambda theta, z: loss(theta, z, method))

    def update(state, k):
        theta, m, v, key = state
        key, subkey = jax.random.split(key)
        z = sample(subkey, batch_size)
        value, gradient = jax.value_and_grad(objective)(theta, z)
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient * gradient
        rate = learning_rates[0] * jnp.where(k < change_after, 1.0,
                                             learning_rates[1] / learning_rates[0])
        direction = (m / (1 - 0.9**(k + 1))) / (jnp.sqrt(v / (1 - 0.999**(k + 1))) + 1e-8)
        theta = theta - rate * direction
        return (theta, m, v, key), (value, theta)

    @jax.jit
    def run_chunk(state, indices):
        return jax.lax.scan(update, state, indices)

    state = (theta0, jnp.zeros_like(theta0), jnp.zeros_like(theta0), jax.random.key(seed))
    # Preserve the original fitting schedules: SCS chunks of 250, DCS one scan.
    chunk_size = min(250, steps) if method == "scs" else steps
    history = []
    for start in range(0, steps, chunk_size):
        stop = min(start + chunk_size, steps)
        state, (values, parameters) = run_chunk(state, jnp.arange(start, stop))
        values, parameters = np.asarray(values), np.asarray(parameters)
        if not np.isfinite(values).all() or not np.isfinite(parameters).all():
            raise FloatingPointError("Non-finite VI loss or parameters")
        history.append(parameters)
        print(f"{method.upper()}: {stop}/{steps} iterations, mean loss {values.mean():.6f}", flush=True)

    history = np.concatenate(history)
    holdout = sample(jax.random.key(seed + 500), holdout_size)
    count = min(average_last, steps)
    candidates = [history[-1], history[-count:].mean(axis=0)]
    candidate_losses = [float(objective(jnp.asarray(theta), holdout)) for theta in candidates]
    if not np.isfinite(candidate_losses).all():
        raise FloatingPointError("Non-finite holdout loss")
    chosen = int(np.argmin(candidate_losses))
    mu, radius, offset = unpack(jnp.asarray(candidates[chosen]), method)
    name = "horizontal_observer" if method == "scs" else "delta"
    parameters = {"mu": np.asarray(mu).tolist(), "radius": float(radius), name: np.asarray(offset).tolist()}
    info = dict(method=method, dimension=D, df=DF, seed=seed, steps=steps,
                batch_size=batch_size, holdout_size=holdout_size, holdout_seed=seed + 500,
                initial_radius=initial_radius, initial_holdout_loss=float(objective(theta0, holdout)),
                candidate_holdout_losses=candidate_losses,
                chosen_candidate=["last", f"last_{count}_parameter_average"][chosen],
                learning_rates=list(learning_rates), learning_rate_change_after_step=change_after,
                adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                objective="KL(pushforward reference || target), up to constants")
    info["fixed_latitude" if method == "scs" else "beta"] = LATITUDE if method == "scs" else 2.0
    return parameters, info


def save_fit(path, parameters, info):
    """Save a new fit separately from the parameters used for the paper figures."""
    path.write_text(json.dumps({"parameters": parameters, "fit": info}, indent=2) + "\n")
    print(f"Saved {path}")
