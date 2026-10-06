"""Target density, transformations and coupling functions."""
from pathlib import Path
import numpy as np
import pandas as pd

# Degrees of freedom of the regression errors.
NU = 4
FOLDER = Path(__file__).resolve().parent

def load_data():
    data = pd.read_csv(FOLDER / "BostonHousing.csv")
    X = data.drop(columns="medv").to_numpy(dtype=float)
    y = data["medv"].to_numpy(dtype=float)
    # Same standardization as StandardScaler; no intercept is added.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y


def log_density(theta, X, y):
    beta, u = theta[:-1], theta[-1]     # u = log(sigma)
    residuals = y - X @ beta
    # Stable form of log(1 + residuals**2 / (NU * exp(2*u))).
    with np.errstate(divide="ignore"):
        terms = 2 * np.log(np.abs(residuals)) - np.log(NU) - 2 * u
    return -0.5 * (NU + 1) * np.logaddexp(0, terms).sum() - len(y) * u


def to_sphere(x, R):
    norm2 = x @ x
    return np.r_[2 * R * x / (R**2 + norm2),
                 (norm2 - R**2) / (R**2 + norm2)]


def from_sphere(z, R):
    return R * z[:-1] / (1 - z[-1])


def sphere_log_density(z, X, y, R):
    d = len(z) - 1
    return log_density(from_sphere(z, R), X, y) - d * np.log1p(-z[-1])


def sphere_log_proposal(z, p, h):
    c = min(float(z @ p), 1.0)
    if c <= 0:
        return -np.inf
    return -len(z) * np.log(c) - (1 - c) * (1 + c) / (2 * h**2 * c**2)


def coupled_proposals(a, b, noise, uniform, h, sphere):
    if sphere:
        tangent = noise - a * (a @ noise) / (a @ a)
        p = a + h * tangent
        p /= np.linalg.norm(p)
        log_ratio = sphere_log_proposal(b, p, h) - sphere_log_proposal(a, p, h)
    else:
        p = a + h * noise
        delta = (a - b) / h
        log_ratio = -noise @ delta - 0.5 * (delta @ delta)

    if np.array_equal(a, b) or np.log(uniform) <= min(0, log_ratio):
        return p, p.copy()

    direction = (a - b) / np.linalg.norm(a - b)
    if sphere:
        q = p - 2 * direction * (direction @ p)
        q /= np.linalg.norm(q)
    else:
        q = b + h * (noise - 2 * direction * (direction @ noise))
    return p, q


def coupled_sampler(x, y, log_target, h, n_steps, rng, sphere=False, R=1):
    a, b = x.copy(), y.copy()
    log_a, log_b = log_target(a), log_target(b)
    accepted_a = accepted_b = 0
    meeting_time = np.inf

    for iteration in range(1, n_steps + 1):
        p, q = coupled_proposals(a, b, rng.normal(size=len(a)), rng.random(), h, sphere)
        log_p = log_target(p)
        log_q = log_p if np.array_equal(p, q) else log_target(q)
        log_u = np.log(rng.random())     # The same MH uniform for both chains.
        if log_u <= min(0, log_p - log_a):
            a, log_a = p, log_p
            accepted_a += 1
        if log_u <= min(0, log_q - log_b):
            b, log_b = q, log_q
            accepted_b += 1
        if not np.isfinite(meeting_time):
            distance = np.linalg.norm(from_sphere(a, R) - from_sphere(b, R)) if sphere else np.linalg.norm(a - b)
            if distance <= 1e-12:
                meeting_time = iteration
        # Continue after meeting: acceptance uses all 20,000 iterations.
    return meeting_time, accepted_a / n_steps, accepted_b / n_steps

