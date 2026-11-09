# Coupling algorithms and variational inference

Sampling has an explicit function for each coupling and map family.

## Method-specific samplers

All samplers start with `(x, y, log_pdf, h, n_steps, rng)`. The initial states
and target density are in the original Euclidean coordinates. Additional map
arguments are keyword-only:

| Sampler | Required map arguments | Proposal |
|---|---|---|
| `coupled_euclidean` | None | Gaussian in Euclidean coordinates |
| `coupled_stereographic` | `mu`, `preconditioning_matrix` | Tangent Gaussian on the sphere |
| `coupled_scs` | `mu`, `preconditioning_matrix`, `horizontal_observer`; optional `latitude=1.1` | Tangent Gaussian, rejecting the dark side |
| `coupled_dcs_ball` | `mu`, `radius`, `delta` | Ball walk with the Mobius map |
| `coupled_dcs_random_walk` | `mu`, `radius`, `delta` | Gaussian walk with the Mobius map |
| `coupled_dcs_matrix_ball` | `mu`, `preconditioning_matrix` | Ball walk with the matrix radial map |
| `coupled_dcs_matrix_random_walk` | `mu`, `preconditioning_matrix` | Gaussian walk with the matrix radial map |

`mu` is a length-d location vector. A preconditioning matrix is a d-by-d
lower-triangular matrix with positive diagonal. For an isotropic stereographic
or SCS map, pass `preconditioning_matrix=R * np.eye(d)`.

SCS latitude must be between 0 and 2. Its horizontal observer must have norm
below `sqrt(latitude * (2 - latitude))`. For Mobius DCS, `radius` is the positive
scale R and `delta` has norm below one. Matrix DCS has its scale in the matrix
and has no `radius` or `delta` argument. Both DCS map families use beta=2.

`h` is a Gaussian standard deviation for Euclidean coupling and a tangent
Gaussian scale for sphere proposals. DCS works in unit-ball coordinates: `h`
is the ball-step radius for ball walk, or the coordinate standard deviation
for Gaussian random walk. In the paper's scalar-radius coordinates, these are
respectively gamma/R and sigma/R. DCS rejects proposals with norm at least one
before evaluating the map; it neither clips them nor resamples them.

For example:

```python
import numpy as np
from generic_algorithms import coupled_scs

result = coupled_scs(
    x, y, log_pdf, h=0.1, n_steps=1000, rng=np.random.default_rng(42),
    mu=np.zeros(d), preconditioning_matrix=np.eye(d),
    horizontal_observer=np.zeros(d), latitude=1.1,
    stop_at_meeting=True, store_chains=True,
)
```

Every sampler supports `stop_at_meeting=True` and `store_chains=False` by
default. Results contain `meeting_time`, `iterations`, both acceptance rates,
and `final_x`, `final_y`. Stored chains include the initial states and are
returned in Euclidean coordinates. Initially identical states meet at time zero;
unmet pairs return `np.inf`. Meetings require exact equality in native
coordinates, and the chains remain together afterwards. Zero-update acceptance
rates are `NaN`; boundary rejections count as rejected updates. These are
simultaneous, zero-lag couplings, rather than a lagged unbiased estimator.

The short single-chain convenience function takes the sampler itself:

```python
from generic_algorithms import sample_chain, coupled_dcs_matrix_random_walk

chain, acceptance = sample_chain(
    coupled_dcs_matrix_random_walk, x, log_pdf, 0.1, 1000,
    np.random.default_rng(42), mu=mu, preconditioning_matrix=L,
)
```

## Run a fixed-parameter experiment

Each sampler has a corresponding `run_<name>_experiment` function in
`run_experiment.py`. For example, `coupled_scs` corresponds to
`run_scs_experiment`, and `coupled_dcs_matrix_random_walk` corresponds to
`run_dcs_matrix_random_walk_experiment`. All seven are exported by the package.

Their first arguments are `(log_pdf, initial_pairs, h, n_steps)`, followed by
exactly the map arguments shown above. `initial_pairs` has shape
`(repetitions, 2, dimension)`. Call each experiment separately, then combine
its result rows if you want one CSV or comparison plot:

```python
from generic_algorithms import (
    run_stereographic_experiment, run_scs_experiment,
    run_dcs_ball_experiment, run_dcs_matrix_random_walk_experiment,
    combine_results, save_results,
)

stereographic = run_stereographic_experiment(
    log_pdf, initial_pairs, h=0.2, n_steps=1000,
    mu=mu, preconditioning_matrix=L,
)
scs = run_scs_experiment(
    log_pdf, initial_pairs, h=0.1, n_steps=1000,
    mu=mu, preconditioning_matrix=L,
    horizontal_observer=observer, latitude=1.1,
)
dcs_ball = run_dcs_ball_experiment(
    log_pdf, initial_pairs, h=0.14, n_steps=1000,
    mu=mu, radius=R, delta=delta,
)
dcs_gaussian = run_dcs_matrix_random_walk_experiment(
    log_pdf, initial_pairs, h=0.05, n_steps=1000,
    mu=mu, preconditioning_matrix=L,
)
rows = combine_results(stereographic, scs, dcs_ball, dcs_gaussian)
save_results(rows, "meeting_times.csv")
```

The example step sizes illustrate the interface; tune each for the target.
The application scripts show complete examples using their own fixed values
or saved fits. For fitted maps, both forms are valid:

```python
scs_fit = parameters["scs"]
rows = run_scs_experiment(
    log_pdf, initial_pairs, 0.1, 1000,
    mu=scs_fit["mu"], preconditioning_matrix=scs_fit["preconditioning_matrix"],
    horizontal_observer=scs_fit["horizontal_observer"], latitude=1.1,
)
# Equivalently: run_scs_experiment(log_pdf, initial_pairs, 0.1, 1000,
#                                 **scs_fit, latitude=1.1)
```

Shared experiment options:

| Argument | Meaning |
|---|---|
| `seed=42` | Base proposal seed; it does not generate the initial pairs. |
| `seed_index` | Stream index for `SeedSequence([seed, seed_index, replicate])`. Defaults are 0 for Euclidean, 1 for stereographic, 2 for SCS, 3 for DCS ball, and 4 for DCS Gaussian. They do not depend on call order. |
| `seed_offset=None` | When supplied, use `default_rng(seed + seed_offset + replicate)` instead. |
| `label` | Label stored in the `method` column; it never selects an algorithm. Defaults are `euclidean`, `stereographic`, `scs`, `dcs`, or `dcs_rw`. |
| `stop_at_meeting=False` | By default, run the full cutoff, including after meeting, for acceptance statistics. |
| `output_csv=None` | Optional CSV for this experiment. Use `save_results` after combining methods for a common CSV. |
| `progress_every=25` | Print progress at this repetition interval. Use `None` or 0 for silence. |

Use distinct seeds or stream indices when comparing several variants of the
same method. For example, two SCS latitudes can have labels `SCS11`/`SCS19`
and distinct `seed_index` values. The application migrations explicitly retain
their previous stream indices and offsets.

Each experiment returns rows with `method`, `replicate`, `met`, `meeting_time`,
`iterations`, `acceptance_rate_chain1`, and `acceptance_rate_chain2`. Censored
times are recorded as -1. `combine_results` interleaves rows by repetition,
preserving the order in which the result groups were supplied. Each individual
experiment now completes before the next one starts.

## Plot saved results

```sh
python generic_algorithms/plot_results.py path/to/meeting_times.csv
python generic_algorithms/plot_results.py path/to/meeting_times.csv --log-scale
```

Or call `plot_results(pd.DataFrame(rows), output_path)` from
`generic_algorithms.plot_results`. The two panels show the histogram of meeting
pairs and the fraction still unmet. The default labels recognize the five names
above, including `dcs_rw`. For other labels, supply
`styles={label: (display_name, color), ...}`. Use `horizon` for the common cutoff
when runs stop at meeting. Meeting time zero is omitted from logarithmic
histogram bins. These plots do not on their own give a stationarity bound for
nonstationary zero-lag starts.

## Implementation and migration

The old `run_experiment(..., method_settings, ...)`, `coupled_sampler(method=...)`,
and `coupled_proposals(proposal=...)` interfaces have been replaced. Select the
function you want directly. All application experiment scripts in this repository
have been migrated. `sample_chain` now takes a sampler function as its first
argument. VI also has method-specific fitting functions, described below; the saved JSON format is preserved.

For old scalar stereographic/SCS fits, convert the scalar scale to
`R * np.eye(d)` when calling the new matrix-based samplers. Existing transformation
utilities still support those old saved maps. Scalar and matrix formulas are
mathematically equivalent for isotropic scales, with possible floating-point
roundoff differences.

The public samplers call the concrete transformations and proposal functions.
`_run_coupling` shares the Metropolis and meeting-time bookkeeping without
selecting a method. Each experiment function contains its own repetition loop and calls its concrete sampler directly.
The dedicated Gaussian, sphere, and ball proposal helpers are still independently
available. Shared mathematical code is kept in one place while the method-specific
arguments and setup are explicit.

Install dependencies with `python -m pip install -r requirements.txt`.
Sampling requires NumPy; VI adds JAX and SciPy; plotting adds pandas and
Matplotlib. JAX uses 64-bit arithmetic when `vi.py` is imported.

## Method-specific VI

The VI functions select their reference distributions and transformations directly:

| Fit function in `vi.py` | Runner exported by the package | Uniform reference | Initial map arguments |
|---|---|---|---|
| `fit_stereographic_vi` | `run_stereographic_vi` | Whole unit sphere S^d | `mu`, `preconditioning_matrix` |
| `fit_scs_vi` | `run_scs_vi` | Bright cap, `z[-1] < latitude - 1` | `mu`, `preconditioning_matrix`, `horizontal_observer`, optional `latitude=1.1` |
| `fit_dcs_vi` | `run_dcs_vi` | Unit ball B(1) | `mu`, `radius`, `delta`, optional `beta=2.0` |
| `fit_dcs_matrix_vi` | `run_dcs_matrix_vi` | Unit ball B(1) | `mu`, `preconditioning_matrix`, optional `beta=2.0` |

All four start with `(log_pdf, dimension)` and take the initial map fields as
keyword arguments. Both the `fit_*_vi` functions and `run_*_vi` runners accept
`tune` to choose the fields Adam updates.

| Method | Fields available to `tune` |
|---|---|
| Stereographic | `mu`, `preconditioning_matrix` |
| SCS | `mu`, `preconditioning_matrix`, `horizontal_observer` |
| Mobius DCS | `mu`, `radius`, `delta`, `beta` |
| Matrix DCS | `mu`, `preconditioning_matrix`, `beta` |

Omitting `tune` or passing `None` fits the map fields with DCS beta held fixed.
Include `"beta"` explicitly to learn the exponent along with the selected fields.
Pass a list or tuple of names to choose a subset, or one name such as
`tune="preconditioning_matrix"`. A vector or matrix is selected as a whole.
Omitted fields keep their supplied physical values during optimization,
holdout selection, and saving. With `tune=()`, the supplied map is evaluated
throughout the requested steps. Duplicate and unsupported names raise an error.

DCS ball walk and Gaussian random walk can share a VI fit since both use the
uniform ball reference. DCS takes `beta=2.0` by default and supports any positive
exponent. SCS latitude sets the cap and is fixed.

The reference samplers are separately callable from `generic_algorithms.vi`:
`sample_uniform_sphere(key, n, dimension)`,
`sample_uniform_scs_cap(key, n, dimension, latitude=1.1)`, and
`sample_uniform_ball(key, n, dimension)`. SCS uses rejection sampling on the
sphere to obtain points uniform on the bright cap. DCS samples a uniform
direction with radius `U**(1/dimension)` to obtain points uniform in volume.
References are independent of the fitted parameters; the SCS latitude is fixed.

For example, fix the SCS location at zero and fit its matrix and observer.

```python
import numpy as np
from generic_algorithms import run_scs_vi

parameters, info, history = run_scs_vi(
    log_pdf_jax, dimension=d,
    mu=np.zeros(d), preconditioning_matrix=np.eye(d),
    horizontal_observer=np.zeros(d), latitude=1.1,
    tune=("preconditioning_matrix", "horizontal_observer"),
    seed=42, steps=4000, batch_size=2048,
    output_json="fitted_parameters.json",
)
```

For a matrix DCS fit with the location fixed, select the matrix.

```python
from generic_algorithms import run_dcs_matrix_vi

parameters, info, history = run_dcs_matrix_vi(
    log_pdf_jax, dimension=d,
    mu=np.zeros(d), preconditioning_matrix=np.sqrt(2.) * np.eye(d),
    tune="preconditioning_matrix",
    seed=43, output_json="fitted_parameters.json",
)
```

Each call returns its own `(parameters, info, history)`. A runner prints the fit
summary, records the starting scale and optional `target_info`, and optionally
saves the map under `label`. The default labels are `stereographic`, `scs`, and
`dcs`. Set `label="SCS19"` to save another SCS fit at latitude 1.9, for example.
A label only names the saved result. Saving one map preserves other maps and
metadata already in the file. Each successful call saves its own result, so a
later fitting failure does not discard earlier completed fits.

For numerical fitting without reporting or saving, import the matching
`fit_*_vi` function directly from `generic_algorithms.vi`. The initial map and
optimizer arguments are the same, and the result is the same triple. Importing
`vi.py` loads JAX and enables 64-bit arithmetic; importing the package, its
runners, or loading saved fits keeps JAX optional.

The target must accept one length-d JAX vector and return one scalar log
density. Use JAX operations and JAX-compatible special functions. Do not convert
the argument or returned value to NumPy or Python float inside the target.
The target normalization constant can be omitted.

All fit functions and runners expose the same optimizer options:

| Argument | Default | Meaning |
|---|---|---|
| `seed` | 0 | Training reference seed; the independent holdout uses seed + 500. |
| `steps` | 4000 | Number of Adam updates. |
| `batch_size` | 2048 | Fresh uniform reference samples per gradient estimate. |
| `learning_rates` | `(0.01, 0.002)` | Rates before and after `change_after`. |
| `change_after` | 3000 | Number of updates using the first rate. |
| `average_last` | 500 | Final unconstrained iterates to average, capped at `steps`. |
| `holdout_size` | 20000 | Independent reference sample count for candidate selection. |
| `chunk_size` | 250 | Updates per compiled JAX scan chunk. |

The runners additionally accept `output_json=None`, `label`, and
`target_info=None`. Application scripts may share a `VI_SETTINGS` dictionary of
these common optimizer controls; each call supplies its own map fields and
reference-specific arguments explicitly. The migrated applications use these direct calls.

## VI objective, history, and parameter conversion

Each method minimizes the same mathematical expectation using its own uniform
reference and map:

```text
E[-log_pdf(T_theta(Z)) - log |Jacobian T_theta(Z)|].
```

This is reverse KL up to constants independent of the fitted parameters. Each
map has its own objective function: `stereographic_reverse_kl`, `scs_reverse_kl`,
`dcs_reverse_kl`, or `dcs_matrix_reverse_kl`. These take
`(theta, samples, log_pdf, dimension)`; SCS additionally takes `latitude`.
They call the corresponding transformations directly. Adam, optimization chunks,
and holdout selection are shared numerical code with no method selection.
Selective fits reconstruct the same map from the free coordinates and the
supplied fixed fields. Only the free coordinates receive Adam updates.

A differentiable target is not sufficient to guarantee finite reverse KL. Choose
a reference/map family with suitable tails for the target. Raw objectives from
different reference families should not be compared as KL values, because their
omitted reference-density constants differ.

The optimizer uses fresh Monte Carlo reference draws at every update. Adam has
beta1=0.9, beta2=0.999, and epsilon=1e-8. After training, the holdout compares the
last iterate with the mean of the final `min(average_last, steps)` unconstrained
vectors. The candidate with the smaller holdout objective supplies the returned
physical map parameters. This comparison does not establish convergence to an
optimum and does not tune an MCMC proposal step size.

`history["loss"][k]` is the loss before update k+1; `history["theta"][k]` is
the parameter vector after that update. Losses can be negative and need not
decrease at each update. Every history contains `steps` losses and parameter
vectors. With the default full fit, their lengths are d+d(d+1)/2 for
stereographic or matrix DCS, 2d+d(d+1)/2 for SCS, and 2d+1 for Mobius DCS.
For a selected subset, `theta` contains only that subset in the field order
shown in the table. `info["theta_slices"]` maps each selected field to its
start and stop columns. `info["tuned_parameters"]` and
`info["fixed_parameters"]` record the chosen fields. Learning DCS beta adds one
log-beta coordinate and a physical-scale `history["beta"]` trace. For example, a matrix-only
fit has d(d+1)/2 columns and a location-only fit has d columns.

For a full fit, use the matching conversion function to inspect its history.

```python
from generic_algorithms.vi import unpack_scs_parameters

parameters_at_update_100 = unpack_scs_parameters(
    history["theta"][99], dimension=d, latitude=1.1,
)
```

The corresponding pairs are `pack_stereographic_parameters` /
`unpack_stereographic_parameters`, `pack_scs_parameters` /
`unpack_scs_parameters`, `pack_dcs_parameters` / `unpack_dcs_parameters`, and
`pack_dcs_matrix_parameters` / `unpack_dcs_matrix_parameters`. Packing takes the
physical map fields shown in the table. Unpacking takes `(theta, dimension)`,
plus `latitude` for SCS. Matrix diagonals are exponentiated, and observer/Mobius
offsets are constrained by a radial tanh parameterization.

`info` records the method, dimension, training controls, initial holdout loss,
candidate holdout losses, and chosen candidate. It also records SCS latitude,
DCS beta, and the matrix parameterization where relevant. Runners additionally
record the initial geometric scale or radius and any application description.

## Load fitted maps for coupling

The saved JSON format is unchanged. Load the required labels, then pass their
fields to the matching sampler or experiment function:

```python
from generic_algorithms import load_parameters, run_scs_experiment

parameters = load_parameters(
    "fitted_parameters.json", dimension=d, methods=["scs"], latitude=1.1,
)
rows = run_scs_experiment(
    log_pdf_numpy, initial_pairs, h=0.1, n_steps=1000,
    **parameters["scs"], latitude=1.1,
)
```

`load_parameters` converts vectors and matrices to NumPy arrays and validates
their dimension, support, and saved SCS latitude without importing JAX. For an
old SCS file without latitude metadata, supply its known original value as
`legacy_scs_latitude`; the loader does not guess. New fits always record it.

For Mobius DCS, pass its fields to `coupled_dcs_ball` or
`coupled_dcs_random_walk`; for matrix DCS, use `coupled_dcs_matrix_ball` or
`coupled_dcs_matrix_random_walk`. Proposal step sizes are chosen separately from
VI, and all couplings still apply the Metropolis correction for the target.

The old `run_vi(..., method_settings, ...)`, `fit_vi(method=...)`,
`sample_reference(method=...)`, and method-dispatched parameter conversion and
objective functions have been replaced by the named functions above.

## Direct calls and parameter validation

The implementation uses ordinary function calls and small named functions; it
contains no `functools.partial` imports or calls. Each experiment has its own
repetition loop, each VI runner calls its fit function directly, and each VI
fit calls its map-specific validator and parameter conversion directly. The
shared numerical helpers handle Adam, holdout selection, Metropolis updates,
and result formatting.

The four validators in `transformations.py` are:

- `validate_stereographic_parameters(parameters, dimension)`
- `validate_scs_parameters(parameters, dimension, latitude=1.1)`
- `validate_dcs_parameters(parameters, dimension)` for scalar-radius Mobius DCS
- `validate_dcs_matrix_parameters(parameters, dimension)` for matrix DCS

They share small checks for a finite location, a valid lower-triangular matrix,
a positive radius, and a bounded offset. The old `validate_parameters(...,
method=...)` function has been removed. Sphere validators also support older
saved scalar scales. The saved-fit loader selects the appropriate validator
from the stored map family and representation.

DCS has separate `dcs_mobius_map` / `dcs_mobius_inverse` and
`dcs_matrix_map` / `dcs_matrix_inverse` functions. Samplers and VI call these
directly. The short `dcs_map` / `dcs_inverse` helpers remain available for
external callers working with either saved representation.

## DCS exponent and the Student-t(d+1) experiment

DCS maps, inverses, coupling runners, and VI support positive `beta`. Matrix
DCS uses `x = mu + L s / (1 - ||s||**beta)**(1/beta)`. Mobius DCS applies its
ball automorphism before the radial transformation. Saved maps record beta.
The loader interprets older DCS maps without beta as using beta=2.

For example, fit a matrix and exponent together with a fixed location.

```python
parameters, info, history = run_dcs_matrix_vi(
    log_pdf_jax, dimension=d, mu=np.zeros(d),
    preconditioning_matrix=np.sqrt(d + 1) * np.eye(d),
    beta=3.0, tune=("preconditioning_matrix", "beta"),
)
```

The optimizer learns `log(beta)` with 0.1 times the usual Adam learning rate.
The Student-t(d+1) experiment uses this joint fit with mu=0. Its
`run_experiment.py` sets the proposal steps explicitly and loads the fitted
beta for sampling. SCS fits only its matrices, stereographic uses radius
sqrt(d), and all five couplings use the saved results in the experiment's
`results/` folder.
