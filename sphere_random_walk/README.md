# Spherical random walk — Section 5.1

The following three Julia files come from [Coupling-on-Sphere](https://github.com/zhihao0539/Coupling-on-Sphere/tree/4e30ceb849872d95c44b8f96abafa6043a98ca56):

- [samplers.jl](samplers.jl): coupling functions required by the simulation.
- [meeting_times_max-reflection_random_walk.jl](meeting_times_max-reflection_random_walk.jl): meeting-time simulation.
- [make_rw_meeting_times_fig.jl](make_rw_meeting_times_fig.jl): plots the saved meeting-time data.

The source repository is unchanged. The plotting script has been moved into this folder, and its input paths now use `data/` instead of `../data/`. No algorithm or parameter changes were made.

## Issues in the preserved Julia code

Review identified two issues in this original copy. The Julia files are retained unchanged as requested; these issues do not affect the separate Python implementation in `generic_algorithms`.

- In `samplers.jl`, `max_reflection_coupling_step!` compares `rand(RNG)` with the negative log proposal-density ratio. The required meeting probability is `min(1, exp(log_ratio))`, so this comparison requires `randexp(RNG)`, or equivalently `log(rand(RNG)) <= min(0, log_ratio)`. The current uniform comparison gives a different coupling kernel and needs correction before using new simulations as maximal-reflection results.
- The simulation uses `h = (d+1)^a`, while `make_rw_meeting_times_fig.jl` reconstructs `hs` with `log.(ds)`. That line needs `log.(ds .+ 1)` to match the simulation. This affects the horizontal coordinates in the third, scaled-step-size plot.

## Original run commands

From the repository root, install the Julia packages and run the simulation:

```sh
cd sphere_random_walk
julia -e 'using Pkg; Pkg.add(["ProgressBars", "JLD", "Plots", "LaTeXStrings"])'
mkdir -p data
julia --threads=auto meeting_times_max-reflection_random_walk.jl
```

Run the plotting script from the same `sphere_random_walk` folder:

```sh
julia make_rw_meeting_times_fig.jl
```

The simulation writes `data/mr_random_walk_meeting_times.jld`. The plotting script reads that file and writes three PNG figures in `sphere_random_walk/`.

The original settings are retained: 1,000,000 repetitions for each of 12 dimensions and 101 step sizes. This is a large simulation; it was not rerun as part of copying the files. Generated data and figures are not included.
