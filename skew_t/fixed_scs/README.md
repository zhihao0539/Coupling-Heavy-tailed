# Skew-t coupling comparison: 1,000 replicates per method

The comparison now contains 4,000 coupled runs: the previous 100 replicates per method were retained and 900 independent initial pairs were added. All four methods use the same 1,000 Euclidean initial pairs. Parameters were frozen before this extension. SCS rejects dark-side proposals directly; it never steps out.

The figure follows the original edited `plot_skew_t_hist_tv_pdf.py` format. An exact copy is supplied as `original_plot_format.py`. The new `plot_meeting_times.py` preserves its 14-by-5-inch two-panel layout, fonts, 34 shared geometric histogram bins, transparency, original colors, grid, spines, and legend convention. DCS is added in purple. The right panel is labelled meeting-time survival. The corresponding graphic is Figure 9 in the supplied draft PDF, although the request refers to Figure 8.

## Results

| Method | Meetings by 1,000,000 | Censoring-aware median | Mean / restricted mean | Bootstrap 95% interval for mean |
|---|---:|---:|---:|---:|
| Stereographic | 558/1,000 | 28,466 | 458,504.41 (restricted) | 428,503.94–488,703.10 |
| Euclidean | 987/1,000 | 15,356 | 55,963.37 (restricted) | 47,540.12–65,343.50 |
| Sub-Cauchy | 1,000/1,000 | 14 | 16.203 | 15.579–16.844 |
| DCS-Ball walk | 1,000/1,000 | 1,341 | 1,691.961 | 1,617.782–1,767.640 |

The median is the first time at which empirical survival reaches 0.5. For DCS, the usual average-of-two-central-order-statistics sample median is 1,342; the first-crossing median used consistently in the table is 1,341. The restricted mean is the average of min(τ, 1,000,000), including censored pairs at the cutoff. Its unrestricted counterpart is not identified for the censored methods by this run. Bootstrap intervals use 10,000 resamples with seed 26092916 and condition on the fitted parameters and proposal sizes.

DCS's mean is 104.42 times SCS's mean, with a paired bootstrap 95% interval of 98.47–110.72. This compares iterations for this target, initialization and frozen tuning. Recorded elapsed times use different batching/backends and must not be used for a computational-efficiency comparison.

## Acceptance rates

| Method | Mean stopped-path rate (figure legend) | Pooled stopped-path rate | Independent stationary estimate |
|---|---:|---:|---:|
| Stereographic | 19.31% | 7.35% | 11.43% |
| Euclidean | 26.50% | 39.03% | 31.17% |
| Sub-Cauchy | 48.16% | 42.07% | 40.64% |
| DCS-Ball walk | 39.71% | 39.55% | 39.96% |

The original plotting script averages each chain's acceptance fraction over its path up to meeting/censoring, then averages all chains equally. The new plot retains that convention. It gives short paths the same weight as long paths. The pooled rate instead divides all accepted proposals by all attempted proposals. Neither statistic is a stationary acceptance estimate because the paths start outside stationarity and stop at meeting. The independent stationary estimates use 40,000 exact target draws. Their Monte Carlo standard errors are 0.141, 0.191, 0.213 and 0.222 percentage points, respectively. Thus the legend's SCS value of 48.2% does not contradict its approximately 40% stationary tuning.

## Target, initialization and parameters

The target is the draft's Azzalini–Capitanio skew-t with d = 100, ν = 2, location zero, scale matrix I and α = (100, −100, 0, …, 0). Its skewing factor is the Student-t CDF T₁₀₂. This retains the user's chosen ν = 2; the DCS reference uses ν = 3. The supplied DCS fit is an independent fit on ν = 2, not the reference experiment's unpublished fitted vectors.

For each pair, two independent unit vectors z on S¹⁰⁰ are sampled uniformly conditional on z₁₀₁ < 0.1. The common Euclidean initial state is x = z₁:₁₀₀/(0.1 − z₁₀₁). Each algorithm uses its own inverse transformation of this same x. The original 100 pairs are preserved exactly in `initial_states_100.npz`; `prepare_initial_states.py` generates the additional 900 pairs using NumPy seed 26092910. `initial_states.npz` stores every Euclidean state and its sub-Cauchy sphere representation.

| Method | Frozen parameters |
|---|---|
| Sub-Cauchy | Observer latitude 1.1, location 0, R = 10/11, h = 0.15; reject dark-side proposals individually |
| Stereographic | R = 10, h = 14 |
| Euclidean | Isotropic Gaussian proposal standard deviation 0.26 |
| DCS-Ball walk | β = 2, R = 1.3878906372792443, γ/R = 0.14, γ = 0.19430468921909422 |

All couplings use maximal reflection proposals and a shared Metropolis uniform. DCS rejects proposals outside its ball separately in each chain. Meeting means exact coalescence; no distance tolerance is used. The experiment has zero lag, no burn-in, no adaptation and a cap of 1,000,000 iterations.

DCS uses F(s) = μ + R Mδ(s)/sqrt(1 − ||Mδ(s)||²) on the unit ball, which is the native B(R) implementation after scaling states and γ by R. All 100 entries of μ and δ are in `dcs_complete_configuration.json`, along with the fitting and tuning settings. The reverse-KL fit used 4,000 Adam steps with batches of 2,048 and seed 26092901. The retained estimate is the average of the final 500 unconstrained parameter vectors, selected against the last iterate on an independent holdout. This extension performs no refitting or retuning. The reduced γ/R = 0.14 was already selected to give approximately 40% stationary acceptance before these 900 new runs.

## Censoring and the plotted curves

The left panel is a histogram conditional on meeting by the cutoff. The 442 stereographic and 13 Euclidean censored runs are excluded only from this histogram. The right panel retains them in the denominator, so its endpoint survival probabilities are 0.442 and 0.013. Censored values are represented as infinity for plotting through the cutoff; this does not assert that their actual meeting times are infinite.

The right panel reports empirical P(τ > t). Two nonstationary zero-lag initial chains do not by themselves provide a total-variation bound to stationarity; a suitable stationary second chain or lagged construction would be needed for that claim.

## Reproduction and validation

`meeting_times.csv` contains all 4,000 observations. Replicates 0–99 come unchanged from `initial_meeting_times_100.csv`. For new replicates 100–999, SCS, DCS and Euclidean use JAX keys obtained by folding the global replicate index into seeds 26092913, 26092923 and 26092943, respectively. All 900 new stereographic observations use the native C++ kernel and independent mt19937_64 streams seeded by splitmix64(26092933 + replicate). No stereographic result was selected between backends based on its outcome or runtime.

The native implementation was used to make the long stereographic runs practical. Its Student-t log CDF agrees with SciPy to 5.7×10⁻¹⁴, its log target agrees with JAX to 1.2×10⁻¹³, and all decisions in 200 deterministic paired transitions matched the JAX reference (maximum state difference 1.2×10⁻¹⁶). `validate_native.py` reproduces these checks. `validate_batch.py` checks the JAX batch/scalar agreement. The prior projection, Jacobian, spherical reflection and uniform-ball coupling validations are retained in `validation.json`. Cross-platform floating-point or C++ normal-generator implementations can change exact random trajectories.

Install Python 3.12 and the packages in `requirements.txt`; the native runner also needs a C++17 compiler (Clang on macOS, g++ on Linux). Run these commands from this directory:

```sh
python plot_meeting_times.py
```

That command immediately reproduces the figure from the saved observations. The full extension workflow is:

```sh
python prepare_initial_states.py
python run_1000_replicates.py --workers 4 --methods scs dcs_bw euclidean
python run_stereographic_native.py --workers 4
python finalize_results.py
python plot_meeting_times.py
```

The repository excludes temporary batch checkpoints. The runners resume any completed outputs present locally. To recompute the 900 new replicates, work in a fresh copy containing the scripts, frozen parameters and the two retained-100 input files, without the `checkpoints/` directory or `stereographic_native_extension.csv`. The first 100 replicates remain fixed inputs to this extension. No refit is necessary. Run `python validate_batch.py` and `python validate_native.py` for the two extension checks. Running `finalize_results.py` after simulation verifies all 4,000 records, retained observations and shared initial arrays, and writes `final_validation.json`. Published-file hashes are in `SHA256SUMS`.

## Suggested figure caption

Meeting times of maximal-reflection couplings targeting the 100-dimensional skew-t distribution with ν = 2 and α = (100, −100, 0, …, 0), using 1,000 common initial pairs. Left: meeting-time histograms conditional on meeting by 10⁶ iterations. Right: empirical meeting-time survival, retaining censored pairs. Sub-Cauchy proposals on the dark side are rejected directly. DCS-Ball walk uses β = 2, a separately fitted Möbius transformation and γ/R = 0.14. SCS and DCS have independent stationary acceptance estimates of 40.6% and 40.0%; legend acceptance rates average the individual paths stopped at meeting or censoring. All SCS and DCS pairs met, while 442 stereographic and 13 Euclidean pairs remained unmet at the cutoff.
