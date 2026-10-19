# Student's t regression experiments

Both applications use the shared coupling and plotting functions in
[`generic_algorithms`](../generic_algorithms/README.md).

- [`boston_housing`](boston_housing/README.md): Boston Housing regression with fixed-parameter stereographic and Euclidean couplings; no VI.
- [`nba_salary`](nba_salary/README.md): NBA salary regression with a matrix stereographic map fitted by VI, plus the saved fit, initial states, and results.

Run either experiment from the repository root:

```sh
python student_t_regression/boston_housing/run_experiment.py
python student_t_regression/nba_salary/run_experiment.py
```

Each folder contains its own target, input data, run settings, and saved results.
See its README for model and parameter details.
