# Generic coupling algorithms and variational inference

These functions take the target log density and tuning parameters as inputs. They do not contain a skew-t target, a regression model, a fixed dimension, data loading, or experiment-specific file paths. The Student-t regression and skew-t experiments use these shared functions; each supplies its own `target.py` and run script.

The six implementation files are:

- `algorithms.py`: maximal-reflection proposal couplings, one shared Metropolis loop, and a single-chain wrapper.
- `transformations.py`: stereographic, sub-Cauchy (SCS), and DCS maps, inverses, and log Jacobians. Sampling and VI use these same formulas.
- `vi.py`: reference sampling, reverse-KL objective, Adam, and one `fit_vi` function for all supported transformation families.
- `run_vi.py`: run VI for selected methods, save the fitted maps, and load them for sampling.
- `run_experiment.py`: a sequential loop over fixed methods and initial pairs, with optional CSV output.
- `plot_results.py`: shared meeting-time histogram and survival plotting, with a CSV command-line entry point.

`__init__.py` makes the folder importable and exposes the shared functions. Importing the runners or loading a saved fit does not load JAX; calling `run_vi` or importing `vi.py` does.

Applications follow the same workflow: define the target in `target.py`, optionally fit maps through `run_vi`, run couplings through `run_experiment`, and plot with `plot_results`. The generic runners are imported functions; the application scripts choose the target and settings. The existing regression and skew-t scripts show this workflow without a separate example application.

## Install and run

From the repository root:

```sh
python -m pip install -r requirements.txt
python student_t_regression/run_experiment.py
python skew_t/run_experiment.py
```

These commands run the settings in each application, including 1,000 repetitions by default; reduce the repetition and iteration counts there for a short trial. Skew-t uses its saved VI maps. To refit them first, run `python skew_t/run_vi.py`. To redraw existing figures without simulation, use the commands in [Plot saved results](#plot-saved-results).

The repository-root `requirements.txt` pins the experiment dependencies. `generic_algorithms/requirements.txt` lists the same libraries without exact version pins for reuse elsewhere. Sampling alone needs NumPy; VI additionally needs JAX and SciPy; plotting additionally needs pandas and Matplotlib. JAX is configured for 64-bit arithmetic when `vi.py` is imported.

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
| `parameters` | Optional fixed map parameters for a transformed method; if omitted, `default_parameters(d, method)` supplies a zero shift and an isotropic scale of `1`. It is unnecessary for `euclidean`. For an isotropic scale `R`, call `default_parameters(d, method, radius=R)`; for a matrix, supply `preconditioning_matrix` as described below. This runner **does not fit** parameters. |
| `latitude` | Optional SCS cap latitude, default `1.1`. For SCS it must lie strictly between `0` and `2`; it has no effect on the other methods. |
| `seed_offset` | Optional integer offset for a method's proposal stream. With it, repetition `rep` uses `np.random.default_rng(seed + seed_offset + rep)`. Use distinct offsets when reproducing application-specific streams. Without it, the runner uses `np.random.SeedSequence([seed, method_index, rep])`, where `method_index` is the method's position in the dictionary. |

For transformed methods, `parameters` includes a length-`d` shift `mu`. Stereographic and SCS maps use `preconditioning_matrix`, a `d`-by-`d` lower-triangular matrix with positive diagonal. `default_parameters(d, method, radius=R)` sets this matrix to `R * I` for those two methods. SCS also has `horizontal_observer`, a length-`d` vector with norm below `sqrt(latitude * (2 - latitude))`. DCS supports either a matrix map with `mu` and `preconditioning_matrix`, or the existing scalar-radius map with `mu`, positive `radius`, and `delta` of norm below `1`. Its default remains the scalar-radius map. Do not combine `preconditioning_matrix` with `radius` or, for DCS, with `delta`. Older saved scalar-radius stereographic and SCS maps remain loadable. Euclidean coupling does not use map parameters. See [Map preconditioning](#map-preconditioning) and [Sampling methods](#sampling-methods).

This minimal use of the shared runner needs only NumPy:

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
python generic_algorithms/plot_results.py skew_t/meeting_times.csv --log-scale
```

The required positional argument is the input CSV path. By default the output has the same path with a `.pdf` suffix; add `--output figure.pdf` to choose another path. Add `--log-scale` when meeting times span several orders of magnitude, as in the skew-t experiment, and `--horizon N` to set the common iteration cutoff explicitly. This redraws the figure without running any samplers. Plotting requires NumPy, pandas, and Matplotlib (included in the repository-root `requirements.txt`).

An application can call the same function directly after running an experiment:

```python
import pandas as pd
from generic_algorithms.plot_results import plot_results

plot_results(pd.DataFrame(rows), "meeting_times.pdf")
# Optional: plot_results(..., log_scale=True, horizon=n_steps)
```

`results` is a DataFrame containing `method`, `meeting_time`, `iterations`, `acceptance_rate_chain1`, and `acceptance_rate_chain2`; `output` is the figure path. The plotter recognizes the four method names used by the shared runner and displays whichever are present. The format has two panels: a histogram among pairs that met, and the fraction of all pairs still un-met at each iteration. Nonnegative meeting times include meetings at initialization; `-1` means censored. Histogram legends show mean acceptance across both chains, excluding undefined rates from zero-update runs; when any pair is censored they also show how many pairs met. The four methods are labelled Stereographic, Euclidean, Sub-Cauchy and DCS-Ball walk.

With `log_scale=True`, the histogram uses 35 logarithmic bins and the survival panel a symlog iteration axis evaluated on a geometric grid. Meetings at time 0 cannot appear on a logarithmic axis and are then left out of the histogram, though they still count in the survival curve and in the met fraction.

The plotter assumes a common iteration cutoff. By default it uses the largest recorded `iterations` as its displayed horizon; pass `horizon` when that may be smaller than the cutoff, for example with `stop_at_meeting=True` when every pair met. Experiments with different censoring cutoffs need a separate survival analysis. Both application run scripts call this function and have no plotting file of their own.

## Map preconditioning

`preconditioning_matrix` is the matrix `L` that stretches the coordinates after the basic projection. For stereographic and SCS, the map has the form `x = mu + L u(z)`, where `u(z)` is the projection before scaling. For matrix DCS, it is `x = mu + L s / sqrt(1 - ||s||²)` for a point `s` in the unit ball. The transformed log density includes `log |det L|`, computed as the sum of the log diagonal entries of `L`; the inverse maps solve `L u = x - mu`. Thus an isotropic `L = R * I` gives the old scalar scale `R`, while a non-diagonal `L` can account for different marginal scales and correlations. The implied scale matrix is `L L.T`; `L` itself is not the covariance matrix.

The implementation requires `L` to be lower triangular with strictly positive diagonal, so it is invertible and its determinant is easy to compute. For example:

```python
parameters = default_parameters(2, "stereographic", radius=1.0)
parameters["preconditioning_matrix"] = np.array([[2.0, 0.0], [0.5, 1.0]])

matrix_dcs = {
    "mu": np.zeros(2),
    "preconditioning_matrix": np.array([[2.0, 0.0], [0.5, 1.0]]),
}
```

The matrix DCS map is the radial map, with no Möbius offset. The existing DCS map with `radius` and `delta` remains available for saved fits and existing experiments. The VI functions fit all entries of the lower triangle for stereographic and SCS. To fit matrix DCS, pass a matrix-form `initial_parameters` dictionary as above; otherwise DCS VI retains its scalar-radius form.

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

For one pair, the lower-level `coupled_sampler` returns `meeting_time`, `iterations`, `acceptance_rate_chain1`, `acceptance_rate_chain2`, `final_x`, and `final_y`. Set `store_chains=True` on this function to also return `chain1` and `chain2`, including initial states, in application coordinates. For repeated experiments, use `run_experiment`, whose result rows are described above.

- Meeting requires exact equality in native coordinates. Identical starting states have meeting time 0; an unmet pair has `np.inf`.
- `stop_at_meeting=True` stops at meeting or the iteration limit. Acceptance rates use the updates actually performed. With zero updates they are `np.nan`.
- `stop_at_meeting=False` runs the full requested length, including after meeting. This is useful for acceptance estimates over a fixed run.
- Both initial states must have finite log density. Proposals with log density `-inf` are rejected; `NaN` or `+inf` raises an error.
- The chains stay exactly together after meeting. No distance tolerance is used. This differs from the original regression script's 1e-12 meeting threshold.
- These are simultaneous, zero-lag couplings. They do not on their own construct a lagged unbiased estimator or a stationarity bound.

For a single chain, `sample_chain(x, log_pdf, h, n_steps, rng, ...)` returns `(chain, acceptance_rate)` using the same transition implementation. It includes the initial state. Saving full chains is optional for coupled runs because it can use substantial memory.

## Run VI from an application

Use the shared `run_vi` function when an application should only define its target
and settings. It calls `fit_vi` for each selected method, prints the fitting
summary, and optionally saves the maps and VI details. The application
[`skew_t/run_vi.py`](../skew_t/run_vi.py) now contains only imports, constants, and
one call to this runner; fitting, saving, and loading functions live here.

```python
from generic_algorithms import run_vi
from skew_t.target import D, NU, log_density_jax

methods = {
    "scs": {"radius": 10 / 11, "seed": 26093001, "chunk_size": 250},
    "dcs": {"radius": NU**0.5, "seed": 26092901, "chunk_size": 4000},
}
parameters, info, history = run_vi(
    log_density_jax, D, methods,
    latitude=1.1,
    steps=4000, batch_size=2048, holdout_size=20000,
    learning_rates=(0.01, 0.002), change_after=3000, average_last=500,
    target_info={"df": NU},
    output_json="skew_t/fitted_parameters.json",
)
```

This example refits the selected maps when executed. To run the configured
skew-t fit from the repository root, use `python skew_t/run_vi.py`.
The generic file is an imported runner, just like `run_experiment.py`; it has no
standalone experiment or target to run by itself.

```text
run_vi(log_pdf, dimension, method_settings, *, output_json=None,
       latitude=1.1, target_info=None, **vi_settings)
```

| Argument | Meaning |
|---|---|
| `log_pdf` | One-vector, scalar-valued JAX target; the same requirements as `fit_vi` below. |
| `dimension` | Dimension of the original Euclidean target. |
| `method_settings` | Dictionary with any of `"stereographic"`, `"scs"`, and `"dcs"` as keys. Each value supplies that method's settings. |
| `radius` inside a method's settings | Convenience setting for an isotropic initial scale, with zero center and offsets; defaults to 1. It initializes `preconditioning_matrix=radius * I` for stereographic and SCS, and a scalar `radius` for DCS. |
| `initial_parameters` inside a method's settings | A complete initial map dictionary, for a custom center, observer, offset, or matrix. Use this instead of the `radius` shortcut. For matrix DCS, supply `mu` and `preconditioning_matrix` without `delta`. |
| `latitude` | Shared SCS latitude, default 1.1. Keep the same value when sampling. |
| `**vi_settings` | Shared `fit_vi` settings, such as `steps`, `seed`, `batch_size`, and learning rates. A setting inside a method's dictionary overrides the shared value. Omitted values use the `fit_vi` defaults below. |
| `target_info` | Optional JSON-compatible model description, such as `{"df": 2}`. Stored under each method's `vi_info[method]["target_info"]`; it does not affect fitting. |
| `output_json` | Optional output path. Existing methods not refitted in this call are preserved. With `None`, no file is written. |

The three returned dictionaries are keyed by method: `parameters["scs"]`,
`info["scs"]`, and `history["scs"]` contain the corresponding `fit_vi` outputs.
The JSON keeps maps under their method names and details under `vi_info`, so it
uses the existing skew-t file format. Histories are returned in memory, not
written into the JSON. `save_parameters(parameters, info, path)` is also
available from `generic_algorithms.run_vi` if saving should happen separately.

### Load the fit for sampling

```python
from generic_algorithms import load_parameters

parameters = load_parameters(
    "skew_t/fitted_parameters.json", dimension=100, methods=("scs", "dcs"),
    latitude=1.1, legacy_scs_latitude=1.1,
)
```

`load_parameters(path, dimension, methods, *, latitude=1.1,
legacy_scs_latitude=None)` loads only the requested maps, converts their vectors
to NumPy arrays, and checks their dimension, admissible parameters, and SCS
latitude. It does not import JAX. Supply these maps as the `parameters` entries
of the generic coupling runner's method settings.

New VI files record their SCS latitude. The original skew-t file has no VI
metadata, so that application explicitly supplies its known historical value
through `legacy_scs_latitude=1.1`. For another old file, supply its actual known
latitude or refit it; the loader will not guess. Existing metadata always takes
precedence over this fallback, and a mismatch with the requested `latitude`
raises an error.

## Tune one map with vi.py

`fit_vi` is the single-map optimizer used by `run_vi`. It supports `"stereographic"`, `"scs"`, and `"dcs"`; Euclidean coupling has no map to fit. It returns fitted values without running MCMC or saving files. Call it directly when you need control over one individual fit; use `run_vi` above for the usual application workflow.

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
| `initial_parameters` | Starting map dictionary. Use `default_parameters(dimension, method, radius=R)` for an isotropic scale `R`, or supply a valid `preconditioning_matrix` for matrix DCS. Supplied vectors and matrices must have the target dimension. | `None`: zero shift/offsets and isotropic scale `1` |
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

- **`parameters`**: the selected physical map parameters, ready for sampling. Every family has a length-`d` NumPy vector `mu`. Stereographic and SCS VI return `preconditioning_matrix`; SCS also has `horizontal_observer`, with norm below `sqrt(latitude * (2 - latitude))`. DCS returns either `preconditioning_matrix` for its matrix radial map or a positive scalar `radius` and `delta` of norm below `1` for its legacy map. `latitude` is passed separately. DCS fixes `beta=2`.
- **`info`**: the method, dimension, training settings, holdout seed, `initial_holdout_loss`, `candidate_holdout_losses`, and `chosen_candidate`. Candidate losses are ordered as `[last_iterate, last_window_average]`; the smaller one determines the returned parameters. SCS also records `latitude`, and DCS records `beta`.
- **`history`**: `loss`, an array of length `steps`, and `theta`, an array of shape `(steps, p)`. For stereographic or matrix DCS, `p=d+d(d+1)/2`; for SCS, `p=2d+d(d+1)/2`; for legacy scalar DCS, `p=2d+1`. `theta` stores the shift, log diagonal and free lower-triangle matrix entries (or log radius for scalar DCS), and any constrained observer or offset variables.

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

This equals reverse KL, `KL(q_theta || target)`, up to constants independent of the fitted parameters. Each update estimates the expectation with a **Monte Carlo (MC) average** over `batch_size` fresh reference points. JAX computes its gradient, and `adam_update` uses moving averages of the gradient and its square to update `theta`. Its fixed constants are `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`. An exponential parameterization keeps matrix diagonal entries (or the legacy DCS radius) positive, and a radial `tanh` parameterization keeps offsets inside their allowed domains.

After training, the independent holdout compares the final iterate with the average of the last `min(average_last, steps)` unconstrained vectors. Averaging happens **before** converting the log diagonal, log radius, and offset variables into physical parameters. The selected candidate is then converted into the returned `parameters` dictionary. The same transformation formulas in `transformations.py` are used for VI and sampling.

### Reading the convergence history

`history["loss"][k]` is the noisy training objective evaluated **before update `k+1`**, whereas `history["theta"][k]` is the parameter vector **after that update**. Thus they are useful as optimization traces, but they are not an objective/parameter pair evaluated at the same iterate. Loss values need not be nonnegative because parameter-independent constants are omitted, and the training loss need not decrease at every update. The omitted reference-density constants differ between map families, so raw losses from stereographic, SCS, and DCS fits should not be compared as KL values.

For a parameter trace, convert an unconstrained row with:

```python
from generic_algorithms.vi import unpack_parameters

parameters_at_update_100 = unpack_parameters(
    jnp.asarray(history["theta"][99]), d, "dcs",
)
```

For SCS, also pass the fitted run's fixed `latitude`. Inspect loss trends and parameter stability; compare the candidate holdout losses with `info["initial_holdout_loss"]`. Holdout selection chooses between two candidates and does not by itself prove convergence to an optimum. For objective values evaluated at selected saved iterates, `sample_reference` and `reverse_kl` can evaluate them on a separate fixed reference sample.

For a matrix DCS trace, call `unpack_parameters(..., matrix_dcs=True)` because `theta` alone does not identify which DCS parameterization produced it.

VI tunes the transformation geometry. It does not tune proposal `h`, minimize meeting time, or alter the target used in the Metropolis correction. The chosen reference family must have finite reverse KL to the target. Differentiability alone is insufficient: a heavy-tailed variational family can have infinite reverse KL to a Gaussian even when a finite MC batch returns a finite value. For constrained targets, first choose appropriate unconstrained coordinates and include their Jacobian in `log_pdf`.

For the skew-t application, [run_vi.py](../skew_t/run_vi.py) takes the JAX target from [target.py](../skew_t/target.py) and supplies the experiment-specific initial values: dimension `100`; SCS latitude `1.1`, initial radius `10/11`, seed `26093001`; DCS initial radius `sqrt(2)`, seed `26092901`. Initial shifts and offsets are zero. It uses the optimizer defaults above, with `chunk_size=4000` for the original DCS scan layout. Small floating-point differences are possible across execution layouts and library versions; see the application README for its output files.

