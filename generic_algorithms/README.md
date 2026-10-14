# Generic coupling algorithms and variational inference

These functions take the target log density and tuning parameters as inputs. They do not contain a skew-t target, a regression model, a fixed dimension, data loading, or experiment-specific file paths. The Student-t regression experiment uses these shared functions; the skew-t folder still runs its original code.

The four implementation files are:

- `algorithms.py`: maximal-reflection proposal couplings, one shared Metropolis loop, and a single-chain wrapper.
- `transformations.py`: stereographic, sub-Cauchy (SCS), and DCS maps, inverses, and log Jacobians. Sampling and VI use these same formulas.
- `vi.py`: reference sampling, reverse-KL objective, Adam, and one `fit_vi` function for all supported transformation families.
- `run_experiment.py`: a sequential loop over fixed methods and initial pairs, with optional CSV output.

`example.py` is a small application. `test_algorithms.py` contains numerical checks. `__init__.py` makes the folder importable; it does not load JAX unless VI is imported.

## Install and run

From the repository root:

```sh
python -m pip install -r generic_algorithms/requirements.txt
python -m generic_algorithms.example
python -m unittest generic_algorithms.test_algorithms -v
```

Sampling alone needs NumPy. VI additionally needs JAX and SciPy. JAX is configured for 64-bit arithmetic when `vi.py` is imported.

## Run a fixed-parameter experiment

`run_experiment` runs the same initial pairs with each requested coupling method. It runs **sequentially**, with no worker setting, VI step, or plotting step. The application supplies its target log density, starting states, fixed method settings, and number of updates. Import it from the repository root with `from generic_algorithms import run_experiment`.

```text
run_experiment(log_pdf, initial_pairs, method_settings, n_steps,
               *, seed=42, stop_at_meeting=False, output_csv=None,
               progress_every=25)
```

| Argument | What to supply | Default |
|---|---|---|
| `log_pdf` | A function accepting one length-`d` NumPy vector and returning one scalar log density in the **original Euclidean coordinates**. It may omit the normalizing constant. The same function is used for every method. | Required |
| `initial_pairs` | A finite array of shape `(repetitions, 2, d)`. `initial_pairs[rep, 0]` and `[rep, 1]` start the two chains in repetition `rep`; every method gets these same states. The runner does not generate them. Each start must also have finite target log density after its method's transformation. | Required |
| `method_settings` | A nonempty dictionary mapping method names (`"euclidean"`, `"stereographic"`, `"scs"`, `"dcs"`) to their fixed settings, described below. Dictionary order determines the order of rows and, without `seed_offset`, the random streams. | Required |
| `n_steps` | Positive integer maximum number of coupled updates **per pair and method**. This is an iteration cutoff, not a time limit. | Required |
| `seed` | Nonnegative integer base seed for reproducible proposals and acceptance decisions. It does not seed or generate `initial_pairs`. | `42` |
| `stop_at_meeting` | If `False`, run all `n_steps`, even after meeting. If `True`, stop at the first meeting; initially identical chains take zero updates. | `False` |
| `output_csv` | File path for the result table. The runner writes it after all repetitions finish, creates parent directories, and replaces an existing file at this path. If `None`, it returns rows without writing a file. | `None` |
| `progress_every` | Positive integer: print progress after every this many **repetitions**, and after the last repetition. Use `None` or `0` to print nothing. | `25` |

Each value in `method_settings` is a small dictionary:

| Field | Meaning |
|---|---|
| `h` | **Required**, finite positive proposal step size. It is the Gaussian standard deviation in Euclidean coordinates for `euclidean`, the tangent-Gaussian scale on the sphere for `stereographic` and `scs`, and the uniform-ball proposal radius in unit-ball coordinates for `dcs`. Its numerical scale therefore differs across methods. |
| `parameters` | Optional fixed map parameters for a transformed method; if omitted, `default_parameters(d, method)` supplies zero shift/offset and radius `1`. It is unnecessary for `euclidean`. For a chosen radius, call `default_parameters(d, method, radius=R)`, then change individual fields if needed. This runner **does not fit** parameters. |
| `latitude` | Optional SCS cap latitude, default `1.1`. For SCS it must lie strictly between `0` and `2`; it has no effect on the other methods. |
| `seed_offset` | Optional integer offset for a method's proposal stream. With it, repetition `rep` uses `np.random.default_rng(seed + seed_offset + rep)`. Use distinct offsets when reproducing application-specific streams. Without it, the runner uses `np.random.SeedSequence([seed, method_index, rep])`, where `method_index` is the method's position in the dictionary. |

For transformed methods, `parameters` includes a length-`d` shift `mu` and positive `radius`. SCS additionally has `horizontal_observer`, a length-`d` vector with norm below `sqrt(latitude * (2 - latitude))`. DCS additionally has `delta`, a length-`d` vector with norm below `1`. `default_parameters` sets the shift and offsets to zero. Euclidean coupling does not use map parameters. See [Sampling methods](#sampling-methods) for the proposal behind each method.

This small example needs only NumPy. Run it from the repository root (for example, save it as `demo.py` and run `python demo.py`):

```python
import numpy as np
from generic_algorithms import default_parameters, run_experiment

d = 2

def log_pdf(x):
    return -0.5 * np.dot(x, x)  # Standard Gaussian target, up to a constant.

initial_pairs = np.random.default_rng(7).normal(size=(3, 2, d))
methods = {
    "stereographic": {
        "h": 0.2,
        "parameters": default_parameters(d, "stereographic", radius=1.0),
    },
    "euclidean": {"h": 0.6},
}
rows = run_experiment(
    log_pdf, initial_pairs, methods, n_steps=100,
    seed=42, output_csv="demo_meeting_times.csv", progress_every=None,
)
print(rows[0])
```

The chosen step sizes only illustrate the interface; tune them for your target. For the Student-t regression application, [its run script](../student_t_regression/run_experiment.py) creates 1,000 pairs in dimension 14 and uses the following fixed settings with `n_steps=20_000`, `seed=42`, and the default `stop_at_meeting=False`:

```python
methods = {
    "stereographic": {
        "h": 0.007,
        "parameters": default_parameters(14, "stereographic", radius=np.sqrt(14)),
        "seed_offset": 100_000,
    },
    "euclidean": {"h": 0.03, "seed_offset": 200_000},
}
```

The function returns a list of dictionaries, one for every method and initial pair, ordered by repetition and then by method. If `output_csv` is given, it writes the same rows with these columns:

| Column | Meaning |
|---|---|
| `method` | Method name from `method_settings`. |
| `replicate` | Zero-based index into `initial_pairs`. |
| `met` | Whether the chains met by the cutoff. Meeting means **exact equality in the method's native coordinates**; no distance tolerance is used. |
| `meeting_time` | First update at which they met; `0` if the two initial states were identical. An unmet run is **censored** and recorded as `-1`, not as a negative meeting time. Report the fraction met and handle censored runs explicitly in summaries. |
| `iterations` | Updates actually performed: normally `n_steps`, or fewer with `stop_at_meeting=True`. |
| `acceptance_rate_chain1`, `acceptance_rate_chain2` | Accepted proposals divided by `iterations` for each chain. If `stop_at_meeting=True` and the initial states coincide, both rates are `NaN` because there were no updates. |

The runner leaves the two chains together after meeting. With its default `stop_at_meeting=False`, both acceptance rates use the full fixed run. Setting `stop_at_meeting=True` changes their denominator to the number of updates performed through meeting. The runner returns no full chains or plot; use `coupled_sampler(..., store_chains=True)` for trajectories and an application plot script for figures.

## Define a target, fit the map, run one coupling with VI

The log density takes one vector and returns one scalar. Its normalizing constant is unnecessary. Data and model settings can be captured in a closure. For VI the function must support JAX differentiation; write it with `jax.numpy` and JAX-compatible special functions.

```python
import numpy as np
import jax.numpy as jnp
from generic_algorithms import coupled_sampler, default_parameters
from generic_algorithms.vi import fit_vi

d, df = 3, 2.0

def log_pdf(x):
    return -(d + df) / 2 * jnp.log1p(jnp.sum(x * x) / df)

parameters, info, history = fit_vi(
    log_pdf, dimension=d, method="dcs",
    initial_parameters=default_parameters(d, "dcs", radius=np.sqrt(df)),
    seed=42,
)

result = coupled_sampler(
    x=np.zeros(d), y=np.ones(d), log_pdf=log_pdf,
    h=0.2, n_steps=5000, rng=np.random.default_rng(100),
    method="dcs", parameters=parameters,
)
print(result["meeting_time"])
```

This is a usage example, not a recommended universal step size. The shorter `example.py` runs 500 VI iterations. A NumPy evaluator for the same log density is faster in long Python sampling loops; JAX is only required for the evaluator supplied to VI.

## Sampling methods

| `method` | Application state and target measure | Native proposal |
|---|---|---|
| `euclidean` | Vector in R^d; Lebesgue density | Gaussian random walk, standard deviation `h` per coordinate |
| `stereographic` | Vector in R^d; Lebesgue density | Same sphere proposal through a stereographic map |
| `scs` | Vector in R^d; Lebesgue density | Same sphere proposal; reject every dark-side proposal |
| `dcs` | Vector in R^d; Lebesgue density | Uniform-ball random walk of radius `h` in the unit-ball coordinates |

Stereographic and SCS use the spherical proposal internally after mapping a Euclidean target to the sphere. `sphere` is not a separate public method. All methods use maximal-reflection proposal coupling and a shared Metropolis uniform. `sphere_log_proposal` omits only its parameter-independent normalizing constant. The shared Metropolis loop applies to these symmetric proposal kernels; it is not an implementation for arbitrary asymmetric proposals.

The DCS family currently implements beta=2, as in the existing experiment. The target and dimension are generic; beta is not a fitted parameter. Its `h` is the normalized ball radius (the experiment's gamma/R), not a Euclidean-coordinate step size. SCS latitude is supplied separately, defaults to 1.1, and stays fixed during VI. There is no stepping-out function. Rejection sampling inside VI only generates uniform reference-cap points.

The [runner section](#run-a-fixed-parameter-experiment) describes the transformed-method parameter fields. Parameter values returned by `fit_vi` can be passed directly to `coupled_sampler`. Existing saved JSON parameters can also be loaded and passed directly; the sampler converts their lists to arrays. To work with native coordinates explicitly, use `to_native`, `from_native`, `native_log_pdf`, and `coupled_proposals`.

## Meeting times and chains

The result contains `meeting_time`, `iterations`, `acceptance_rate_chain1`, `acceptance_rate_chain2`, `final_x`, and `final_y`. Set `store_chains=True` to also return `chain1` and `chain2`, including initial states, in application coordinates.

- Meeting requires exact equality in native coordinates. Identical starting states have meeting time 0; an unmet pair has `np.inf`.
- `stop_at_meeting=True` stops at meeting or the iteration limit. Acceptance rates use the updates actually performed. With zero updates they are `np.nan`.
- `stop_at_meeting=False` runs the full requested length, including after meeting. This is useful for acceptance estimates over a fixed run.
- Both initial states must have finite log density. Proposals with log density `-inf` are rejected; `NaN` or `+inf` raises an error.
- The chains stay exactly together after meeting. No distance tolerance is used. This differs from the original regression script's 1e-12 meeting threshold.
- These are simultaneous, zero-lag couplings. They do not on their own construct a lagged unbiased estimator or a stationarity bound.

For a single chain, `sample_chain(x, log_pdf, h, n_steps, rng, ...)` returns `(chain, acceptance_rate)` using the same transition implementation. It includes the initial state. Saving full chains is optional for coupled runs because it can use substantial memory.

## What the VI function does

`fit_vi` supports `scs`, `dcs`, and `stereographic`. It minimizes

```text
E[-log_pdf(T_theta(Z)) - log |Jacobian T_theta(Z)|],
```

which is reverse KL up to constants. Reference `Z` is uniform on the bright cap, unit ball, or whole sphere, respectively. All transformations are defined once in `transformations.py`; VI calls them with the JAX backend.

Each update uses fresh reference samples and an automatic gradient. `adam_update` contains the Adam formula explicitly. Positive radii use an exponential parameterization; constrained offsets use a radial tanh parameterization. The defaults retain the earlier tuning schedule:

| Setting | Default |
|---|---:|
| Iterations | 4,000 |
| Samples per iteration | 2,048 |
| Learning rate | 0.01, then 0.002 after 3,000 updates |
| Adam beta1, beta2, epsilon | 0.9, 0.999, 1e-8 |
| Parameter averaging | Last 500 unconstrained vectors |
| Independent selection holdout | 20,000 points, seed = training seed + 500 |

The holdout selects between the final iterate and the average of the last 500 unconstrained vectors. Returns are `(parameters, info, history)`: physical parameters for sampling, settings and holdout losses, and per-update `loss` and unconstrained `theta`. No files are saved automatically. Convergence plots can use `history['loss']`; parameter histories can be decoded with `unpack_parameters`.

VI tunes the transformation geometry. It does not tune proposal `h`, minimize meeting time, or alter the target used in the Metropolis correction. The chosen reference family must have finite reverse KL to the target. Differentiability alone is insufficient: for example, a heavy-tailed variational family can have infinite reverse KL to a Gaussian even when a finite Monte Carlo batch returns a finite value. For constrained targets, first choose appropriate unconstrained coordinates and include their Jacobian in `log_pdf`.

To reproduce the existing skew-t tuning, the application supplies its own log density and these settings: d=100; SCS latitude=1.1, initial radius=10/11, seed=26093001; DCS initial radius=sqrt(2), seed=26092901. Initial shifts and offsets are zero. The optimizer defaults above match the fitting settings; `chunk_size=4000` matches the original DCS scan layout. Small floating-point differences are possible across execution layouts and library versions. The new folder does not load or overwrite the paper's fitted parameters.

## Checks

The tests cover maps and inverses at several dimensions, Jacobians against numerical derivatives, both proposal marginals against independent draws, Gaussian maximal meeting probability, domain rejection, exact meeting and sticking, known Student-t reference families, VI gradients, and fitting/reusing the returned parameters across different dimensions. Local compatibility checks also compare the maps, proposal formulas, transformed log densities, and a 50-iteration VI replay with the existing experiment implementations.
