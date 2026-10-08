# Variational inference for the skew-t experiment

- [vi_functions.py](vi_functions.py): target density, transformation maps, reference sampling and Adam optimization.
- [fit_scs.py](fit_scs.py): SCS fitting settings and run code.
- [fit_dcs.py](fit_dcs.py): DCS fitting settings and run code.

From the repository root:

```sh
python -m pip install -r skew_t/vi/requirements.txt
python skew_t/vi/fit_scs.py
python skew_t/vi/fit_dcs.py
```

Each script saves one file in this folder: `scs_fit.json` or `dcs_fit.json`. The `parameters` object uses the same fields as the corresponding `scs` or `dcs` entry in [fitted_parameters.json](../fitted_parameters.json). To use a new fit, replace that entry with the new `parameters` object. The scripts do not overwrite the saved experiment parameters or meeting-time results.

The functions are extracted and reorganized from the original fitting implementation. They retain the 100-dimensional skew-t target with ν = 2 and α = (100, −100, 0, …, 0), parameterizations, random seeds and optimizer settings:

| Setting | SCS | DCS |
|---|---|---|
| Reference distribution | Uniform bright spherical cap | Uniform unit ball |
| Fixed map parameter | Latitude 1.1 | β = 2 |
| Initial radius | 10/11 | √2 |
| Initial location and offset | Zero | Zero |
| Fitted parameters | Location, radius, horizontal observer | Location, radius, Möbius parameter |
| Random seed | 26093001 | 26092901 |
| Adam iterations | 4,000 | 4,000 |
| Samples per iteration | 2,048 | 2,048 |

Both minimize the reverse KL divergence from the transformed reference distribution to the target. Up to constants, the Monte Carlo objective is the negative average of the target log density plus the transformation's log Jacobian. Adam uses β₁ = 0.9, β₂ = 0.999 and ε = 10⁻⁸. The learning rate is 0.01 for 3,000 iterations, then 0.002. An independent 20,000-sample holdout selects between the final iterate and the average of the last 500 unconstrained parameter vectors; its seed is the fitting seed plus 500. JAX uses 64-bit arithmetic. Small numerical differences between platforms or library versions are possible.

This folder fits the transformation parameters. Proposal step sizes were tuned separately toward 40% stationary acceptance and remain in [run_experiment.py](../run_experiment.py): SCS h = 0.08 and DCS γ/R = 0.14. The meeting-time sampler continues to reject SCS dark-side proposals directly, without stepping out. Rejection sampling here only generates IID reference-cap samples for VI.

The fitting procedures follow the reverse-KL approaches in [Sub-Cauchy Sampling, Section 3](https://arxiv.org/abs/2601.11066) and [Diffeomorphic MCMC, Appendix D](https://arxiv.org/abs/2608.04284), using the draft's ν = 2 target.
