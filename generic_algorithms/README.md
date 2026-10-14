# Generic coupling algorithms and variational inference

These functions take the target log density and tuning parameters as inputs. They do not contain a skew-t target, a regression model, a fixed dimension, data loading, or experiment-specific file paths. The Student-t regression sampler and the skew-t VI script use these shared functions.

The five implementation files are:

- `algorithms.py`: maximal-reflection proposal couplings, one shared Metropolis loop, and a single-chain wrapper.
- `transformations.py`: stereographic, sub-Cauchy (SCS), and DCS maps, inverses, and log Jacobians. Sampling and VI use these same formulas.
- `vi.py`: reference sampling, reverse-KL objective, Adam, and one `fit_vi` function for all supported transformation families.
- `run_experiment.py`: a sequential loop over fixed methods and initial pairs, with optional CSV output.
- `plot_results.py`: shared meeting-time histogram and survival plotting, with a CSV command-line entry point.

`example.py` is a small application. `test_algorithms.py` contains numerical checks. `__init__.py` makes the folder importable; it does not load JAX unless VI is imported.

## Install and run

From the repository root:

```sh
python -m pip install -r generic_algorithms/requirements.txt
python -m generic_algorithms.example
python -m unittest generic_algorithms.test_algorithms -v
```

Sampling alone needs NumPy. VI additionally needs JAX and SciPy. Plotting additionally needs pandas and Matplotlib, available with `python -m pip install pandas matplotlib`. JAX is configured for 64-bit arithmetic when `vi.py` is imported.

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

The runner leaves the two chains together after meeting. With its default `stop_at_meeting=False`, both acceptance rates use the full fixed run. Setting `stop_at_meeting=True` changes their denominator to the number of updates performed through meeting. The runner returns no full chains or plot; use `coupled_sampler(..., store_chains=True)` for trajectories and the shared `plot_results` function for figures.

## Plot saved results

From the repository root, plot a CSV produced by the shared runner:

```sh
python generic_algorithms/plot_results.py student_t_regression/meeting_times.csv
```

The required positional argument is the input CSV path. By default the output has the same path with a `.pdf` suffix; add `--output figure.pdf` to choose another path. This redraws the figure without running any samplers. Plotting requires NumPy, pandas, and Matplotlib (included in the repository-root `requirements.txt`).

An application can call the same function directly after running an experiment:

```python
import pandas as pd
from generic_algorithms.plot_results import plot_results

plot_results(pd.DataFrame(rows), "meeting_times.pdf")
```

`results` is a DataFrame containing `method`, `meeting_time`, `iterations`, `acceptance_rate_chain1`, and `acceptance_rate_chain2`; `output` is the figure path. The plotter recognizes the four method names used by the shared runner and displays whichever are present. The format has two panels: a histogram among pairs that met, and the fraction of all pairs still un-met at each iteration. Nonnegative meeting times include meetings at initialization; `-1` means censored. Histogram legends show mean acceptance across both chains, excluding undefined rates from zero-update runs.

The plotter assumes a common iteration cutoff and uses the largest recorded `iterations` as its displayed horizon. Experiments with different censoring cutoffs need a separate survival analysis. The Student-t regression run calls this function automatically and has no application-specific plotting file. Its saved figure format is retained.

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

## Tune parameters with vi.py

Use `fit_vi` to choose the **map parameters** before running a coupling experiment. It supports `"stereographic"`, `"scs"`, and `"dcs"`; Euclidean coupling has no map to fit. It returns fitted values without running MCMC or saving files. The application script [skew_t/run_vi.py](../skew_t/run_vi.py) shows how the skew-t experiment calls this function for SCS and DCS.

Import it from the repository root:

```python
from generic_algorithms.vi import fit_vi
```

The target must accept **one length-`d` vector** in the original Euclidean coordinates and return **one scalar log density**. Its normalizing constant can be omitted. Use `jax.numpy` and JAX-compatible special functions, because JAX differentiates through this evaluator. Do not convert its argument or return value to a NumPy array or Python `float` inside the function. Data and fixed model settings can be captured in a closure.

For example, this fits the DCS map to a three-dimensional Student-t target with two degrees of freedom:

```python
import numpy as np
import jax.numpy as jnp
from generic_algorithms import default_parameters
from generic_algorithms.vi import fit_vi

d, df = 3, 2.0

def log_pdf_jax(x):
    return -(d + df) / 2 * jnp.log1p(jnp.sum(x * x) / df)

parameters, info, history = fit_vi(
    log_pdf_jax,
    dimension=d,
    method="dcs",
    initial_parameters=default_parameters(d, "dcs", radius=np.sqrt(df)),
    seed=42,
)
print(parameters)
print(info["chosen_candidate"])
```

The full interface is:

```text
fit_vi(log_pdf, dimension, method='scs', *, initial_parameters=None,
       latitude=1.1, seed=0, steps=4000, batch_size=2048,
       learning_rates=(0.01, 0.002), change_after=3000,
       average_last=500, holdout_size=20000, chunk_size=250)
```

| Argument | Meaning | Default |
|---|---|---|
| `log_pdf` | Differentiable scalar JAX log density described above. | Required |
| `dimension` | Positive integer dimension of the target vector. | Required |
| `method` | Map family: `"stereographic"`, `"scs"`, or `"dcs"`. | `"scs"` |
| `initial_parameters` | Starting map dictionary. Use `default_parameters(dimension, method, radius=R)` for zero shift and offsets with radius `R`. Supplied vectors must have length `dimension` and satisfy the constraints below. | `None`: zero shift/offsets, radius `1` |
| `latitude` | Fixed SCS cap latitude, strictly between `0` and `2`. Use the same value later in sampling. It is not optimized and has no effect on the other families. | `1.1` |
| `seed` | Random seed for training samples. The selection holdout uses `seed + 500`. | `0` |
| `steps` | Positive integer number of Adam updates. | `4000` |
| `batch_size` | Positive integer number of fresh reference samples in each Monte Carlo gradient. Larger batches reduce sampling noise but require more computation and memory. | `2048` |
| `learning_rates` | Two positive learning rates: the first before the change, the second afterwards. | `(0.01, 0.002)` |
| `change_after` | Number of updates using the first rate. With `0`, every update uses the second; with a value at least `steps`, every update uses the first. | `3000` |
| `average_last` | Number of final **unconstrained parameter vectors** to average for one candidate solution; capped at `steps`. This does not average sampled chains. | `500` |
| `holdout_size` | Positive integer number of independent reference points used to compare the final iterate and averaged candidate. | `20000` |
| `chunk_size` | Positive integer number of updates processed in each compiled JAX scan. This controls execution grouping, not the total number of updates or the batch size. | `250` |

`fit_vi` returns three objects:

- **`parameters`**: the selected physical map parameters, ready for sampling. Every family has a length-`d` NumPy vector `mu` and a positive Python scalar `radius`. SCS also has `horizontal_observer`, with norm below `sqrt(latitude * (2 - latitude))`; DCS has `delta`, with norm below `1`. `latitude` is passed separately. The DCS implementation fixes `beta=2`.
- **`info`**: the method, dimension, training settings, holdout seed, `initial_holdout_loss`, `candidate_holdout_losses`, and `chosen_candidate`. Candidate losses are ordered as `[last_iterate, last_window_average]`; the smaller one determines the returned parameters. SCS also records `latitude`, and DCS records `beta`.
- **`history`**: `loss`, an array of length `steps`, and `theta`, an array of shape `(steps, p)`, where `p=d+1` for stereographic or `p=2*d+1` for SCS/DCS. `theta` stores unconstrained variables: shift, log radius, and, where needed, the vector used to parameterize the constrained offset.

To reuse a fitted map, place `parameters` in the chosen method's settings. The following continues the example above using a NumPy evaluator of the same target for the sampling loop:

```python
from generic_algorithms import run_experiment

def log_pdf_numpy(x):
    return -(d + df) / 2 * np.log1p(np.sum(x * x) / df)

methods = {"dcs": {"parameters": parameters, "h": 0.2}}
initial_pairs = np.random.default_rng(7).normal(size=(3, 2, d))
rows = run_experiment(
    log_pdf_numpy, initial_pairs, methods, n_steps=100,
    seed=42, progress_every=None,
)
```

Here `h=0.2` only illustrates the interface: choose the proposal step size separately for your target. For SCS, include `"latitude": latitude` in the method settings, using the value supplied to `fit_vi`. The [runner guide](#run-a-fixed-parameter-experiment) explains the remaining experiment settings and CSV output.

### Objective, Monte Carlo estimates, and Adam

Let `T_theta` map a reference point `Z` into the target's Euclidean coordinates. The reference is uniform on the whole sphere for stereographic, the bright cap for SCS, or the unit ball for DCS. The induced distribution of `T_theta(Z)` is the variational distribution. The fitted objective is

```text
E[-log_pdf(T_theta(Z)) - log |Jacobian T_theta(Z)|].
```

This equals reverse KL, `KL(q_theta || target)`, up to constants independent of the fitted parameters. Each update estimates the expectation with a **Monte Carlo (MC) average** over `batch_size` fresh reference points. JAX computes its gradient, and `adam_update` uses moving averages of the gradient and its square to update `theta`. Its fixed constants are `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`. An exponential parameterization keeps the radius positive, and a radial `tanh` parameterization keeps offsets inside their allowed domains.

After training, the independent holdout compares the final iterate with the average of the last `min(average_last, steps)` unconstrained vectors. Averaging happens **before** converting the log radius and offset variables into physical parameters. The selected candidate is then converted into the returned `parameters` dictionary. The same transformation formulas in `transformations.py` are used for VI and sampling.

### Reading the convergence history

`history["loss"][k]` is the noisy training objective evaluated **before update `k+1`**, whereas `history["theta"][k]` is the parameter vector **after that update**. Thus they are useful as optimization traces, but they are not an objective/parameter pair evaluated at the same iterate. Loss values need not be nonnegative because parameter-independent constants are omitted, and the training loss need not decrease at every update.

For a parameter trace, convert an unconstrained row with:

```python
from generic_algorithms.vi import unpack_parameters

parameters_at_update_100 = unpack_parameters(
    jnp.asarray(history["theta"][99]), d, "dcs",
)
```

For SCS, also pass the fitted run's fixed `latitude`. Inspect loss trends and parameter stability; compare the candidate holdout losses with `info["initial_holdout_loss"]`. Holdout selection chooses between two candidates and does not by itself prove convergence to an optimum. For objective values evaluated at selected saved iterates, `sample_reference` and `reverse_kl` can evaluate them on a separate fixed reference sample.

VI tunes the transformation geometry. It does not tune proposal `h`, minimize meeting time, or alter the target used in the Metropolis correction. The chosen reference family must have finite reverse KL to the target. Differentiability alone is insufficient: a heavy-tailed variational family can have infinite reverse KL to a Gaussian even when a finite MC batch returns a finite value. For constrained targets, first choose appropriate unconstrained coordinates and include their Jacobian in `log_pdf`.

For the skew-t application, [run_vi.py](../skew_t/run_vi.py) supplies the target and experiment-specific initial values: dimension `100`; SCS latitude `1.1`, initial radius `10/11`, seed `26093001`; DCS initial radius `sqrt(2)`, seed `26092901`. Initial shifts and offsets are zero. It uses the optimizer defaults above, with `chunk_size=4000` for the original DCS scan layout. Small floating-point differences are possible across execution layouts and library versions; see the application README for its output files.

## Checks

The tests cover maps and inverses at several dimensions, Jacobians against numerical derivatives, both proposal marginals against independent draws, Gaussian maximal meeting probability, domain rejection, exact meeting and sticking, known Student-t reference families, VI gradients, and fitting/reusing the returned parameters across different dimensions. Local compatibility checks also compare the maps, proposal formulas, transformed log densities, and a 50-iteration VI replay with the existing experiment implementations.
