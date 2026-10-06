"""Skew-t coupling: functions, simulation, then the two-panel figure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# Change these settings to run a new experiment.
RUN_SIMULATION = False             # False: plot the saved 1,000 repetitions.
N_REP = 1000
MAX_STEPS = 1000000
D = 100
NU = 2
LATITUDE = 1.1
SEED = 42
STEPS = {"stereographic": 14.0, "euclidean": 0.26, "scs": 0.08, "dcs_bw": 0.14}
FOLDER = Path(__file__).resolve().parent


def load_parameters():
    parameters = json.loads((FOLDER / "fitted_parameters.json").read_text())
    for values in parameters.values():
        for name, value in values.items():
            if isinstance(value, list):
                values[name] = np.asarray(value)
    return parameters


def log_density(x):
    norm2 = x @ x
    skew = 100 * (x[0] - x[1]) * np.sqrt((D + NU) / (NU + norm2))
    return -(D + NU) / 2 * np.log1p(norm2 / NU) + t.logcdf(skew, D + NU)


def to_sphere(x, R=10):
    norm2 = x @ x
    return np.r_[2 * R * x / (norm2 + R**2), (norm2 - R**2) / (norm2 + R**2)]


def scs_map(z, parameters):
    mu, radius, observer = parameters["mu"], parameters["radius"], parameters["horizontal_observer"]
    height = z[-1]
    gap = LATITUDE - 1 - height
    x = mu + radius * (LATITUDE * z[:-1] - (1 + height) * observer) / gap
    log_jacobian = D * np.log(radius * LATITUDE) + np.log(1 - (LATITUDE - 1) * height - observer @ z[:-1]) - (D + 1) * np.log(gap)
    return x, log_jacobian


def scs_inverse(x, parameters):
    mu, radius, observer = parameters["mu"], parameters["radius"], parameters["horizontal_observer"]
    b = LATITUDE - 1
    a = (x - mu) / radius - observer
    aa = a @ a + LATITUDE**2
    half_b = a @ observer - LATITUDE * b
    cc = observer @ observer + b**2 - 1
    root = np.sqrt(half_b**2 - aa * cc)
    m = -cc / (root + half_b) if half_b > 0 else (-half_b + root) / aa
    return np.r_[m * a + observer, b - LATITUDE * m]


def mobius(s, delta):
    norm2, dot, delta2 = s @ s, s @ delta, delta @ delta
    return ((1 - delta2) * s + (1 + 2 * dot + norm2) * delta) / (1 + 2 * dot + delta2 * norm2)


def dcs_map(s, parameters):
    mu, radius, delta = parameters["mu"], parameters["radius"], parameters["delta"]
    norm2, delta2 = s @ s, delta @ delta
    denominator = 1 + 2 * (s @ delta) + delta2 * norm2
    gap = (1 - delta2) * (1 - norm2) / denominator
    x = mu + radius * mobius(s, delta) / np.sqrt(gap)
    log_jacobian = D * np.log(radius) + D * np.log1p(-delta2) - D * np.log(denominator) - (1 + D / 2) * np.log(gap)
    return x, log_jacobian


def dcs_inverse(x, parameters):
    u = (x - parameters["mu"]) / parameters["radius"]
    return mobius(u / np.sqrt(1 + u @ u), -parameters["delta"])


def native_log_density(z, method, parameters):
    if method == "scs":
        if z[-1] >= LATITUDE - 1:
            return -np.inf          # Reject the dark side; no stepping out.
        x, log_jacobian = scs_map(z, parameters["scs"])
    elif method == "dcs_bw":
        if z @ z >= 1:
            return -np.inf          # Reject proposals outside the unit ball.
        x, log_jacobian = dcs_map(z, parameters["dcs"])
    elif method == "stereographic":
        if z[-1] >= 1:
            return -np.inf
        x = 10 * z[:-1] / (1 - z[-1])
        log_jacobian = -D * np.log1p(-z[-1])
    else:
        x, log_jacobian = z, 0
    return log_density(x) + log_jacobian


def sphere_log_proposal(z, p, h):
    c = min(float(z @ p), 1.0)
    if c <= 0:
        return -np.inf
    return -(D + 1) * np.log(c) - (1 - c) * (1 + c) / (2 * h**2 * c**2)


def coupled_proposals(a, b, noise, uniform, radius_uniform, h, method):
    sphere = method in ("stereographic", "scs")
    if sphere:
        p = a + h * (noise - a * (a @ noise) / (a @ a))
        p /= np.linalg.norm(p)
        log_ratio = sphere_log_proposal(b, p, h) - sphere_log_proposal(a, p, h)
        common = np.log(uniform) <= min(0, log_ratio)
    else:
        if method == "dcs_bw":
            move = h * noise / np.linalg.norm(noise) * radius_uniform**(1 / D)
        else:
            move = h * noise
        p = a + move
        if method == "dcs_bw":
            common = np.sum((p - b)**2) <= h**2
        else:
            log_ratio = (move @ move - np.sum((p - b)**2)) / (2 * h**2)
            common = np.log(uniform) <= min(0, log_ratio)
    if np.array_equal(a, b) or common:
        return p, p.copy()
    direction = (a - b) / np.linalg.norm(a - b)
    if sphere:
        q = p - 2 * direction * (direction @ p)
        q /= np.linalg.norm(q)
    else:
        q = b + move - 2 * direction * (direction @ move)
    return p, q


def coupled_sampler(a, b, method, parameters, max_steps, rng):
    a, b = a.copy(), b.copy()
    log_a = native_log_density(a, method, parameters)
    log_b = native_log_density(b, method, parameters)
    accepted_a = accepted_b = 0
    for iteration in range(1, max_steps + 1):
        p, q = coupled_proposals(a, b, rng.normal(size=len(a)), rng.random(),
                                 rng.random(), STEPS[method], method)
        log_p = native_log_density(p, method, parameters)
        log_q = log_p if np.array_equal(p, q) else native_log_density(q, method, parameters)
        log_u = np.log(rng.random())
        if log_u <= min(0, log_p - log_a):
            a, log_a = p, log_p
            accepted_a += 1
        if log_u <= min(0, log_q - log_b):
            b, log_b = q, log_q
            accepted_b += 1
        if np.array_equal(a, b):
            return iteration, iteration, accepted_a / iteration, accepted_b / iteration
    return np.inf, max_steps, accepted_a / max_steps, accepted_b / max_steps


def run_experiment(n_rep=N_REP, max_steps=MAX_STEPS):
    parameters = load_parameters()
    starts = np.load(FOLDER / "initial_states.npz")["initial_states"]
    if not 1 <= n_rep <= len(starts):
        raise ValueError("Choose between 1 and 1,000 saved initial pairs.")
    rows = []
    for rep, (x, y) in enumerate(starts[:n_rep]):
        for index, method in enumerate(STEPS):
            if method == "scs":
                a, b = scs_inverse(x, parameters["scs"]), scs_inverse(y, parameters["scs"])
            elif method == "dcs_bw":
                a, b = dcs_inverse(x, parameters["dcs"]), dcs_inverse(y, parameters["dcs"])
            elif method == "stereographic":
                a, b = to_sphere(x), to_sphere(y)
            else:
                a, b = x, y
            rng = np.random.default_rng(SEED + rep + (index + 1) * 100000)
            tau, iterations, acc1, acc2 = coupled_sampler(a, b, method, parameters, max_steps, rng)
            rows.append(dict(method=method, replicate=rep, met=np.isfinite(tau),
                             meeting_time=tau if np.isfinite(tau) else -1,
                             iterations=iterations, acceptance_rate_chain1=acc1,
                             acceptance_rate_chain2=acc2, cutoff=max_steps))
        if (rep + 1) % 10 == 0:
            print(f"Finished {rep + 1}/{n_rep} repetitions", flush=True)
    return pd.DataFrame(rows)


def plot_results(results, output=FOLDER / "meeting_times.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    finite = results.loc[results.meeting_time > 0, "meeting_time"].to_numpy()
    cutoff = int(results["cutoff"].max()) if "cutoff" in results else MAX_STEPS
    lower = max(1, finite.min()) if len(finite) else 1
    upper = max(lower * 1.01, finite.max() * 1.001) if len(finite) else max(cutoff, 2)
    bins = np.geomspace(lower, upper, 35)
    iters = np.r_[0, np.unique(np.geomspace(1, cutoff, 1600).astype(int))]
    methods = [("stereographic", "Stereographic", "tab:blue"),
               ("euclidean", "Euclidean", "tab:orange"),
               ("scs", "Sub-Cauchy (VI)", "tab:green"),
               ("dcs_bw", "DCS-Ball walk (VI)", "tab:purple")]
    for method, label, color in methods:
        selected = results[results.method == method]
        tau = np.where(selected.meeting_time > 0, selected.meeting_time, np.inf)
        finite_tau = tau[np.isfinite(tau)]
        acceptance = selected[["acceptance_rate_chain1", "acceptance_rate_chain2"]].to_numpy().mean()
        if len(finite_tau):
            axes[0].hist(finite_tau, bins=bins, weights=np.ones(len(finite_tau)) / len(finite_tau),
                         color=color, alpha=0.7, edgecolor="black", linewidth=0.5,
                         label=f"{label}, met={len(finite_tau)}/{len(tau)}, acc={acceptance:.3f}")
        survival = 1 - np.searchsorted(np.sort(tau), iters, side="right") / len(tau)
        axes[1].plot(iters, survival, color=color, linewidth=1.8, label=label)
        print(f"{label}: met {len(finite_tau)}/{len(tau)}, mean min(tau, cutoff)={np.minimum(tau, cutoff).mean():.3f}, acc={acceptance:.3%}")
    axes[0].set(xscale="log", xlabel="Meeting time", ylabel="Probability", title="Finite meeting times (met pairs only)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set(xlim=(0, cutoff), ylim=(-0.015, 1.015), xlabel="Iteration $t$",
                ylabel=r"Empirical survival $\widehat{\mathbb{P}}(\tau > t)$", title="Meeting-time survival")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    if RUN_SIMULATION:
        results = run_experiment()
        results.to_csv(FOLDER / "new_meeting_times.csv", index=False)
        plot_results(results, FOLDER / "new_meeting_times.pdf")
    else:
        results = pd.read_csv(FOLDER / "meeting_times.csv")
        plot_results(results)
