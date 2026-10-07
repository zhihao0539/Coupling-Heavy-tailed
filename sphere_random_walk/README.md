# Spherical random walk — Section 5.1

The following three Julia files are copied unchanged from [Coupling-on-Sphere](https://github.com/zhihao0539/Coupling-on-Sphere/tree/4e30ceb849872d95c44b8f96abafa6043a98ca56):

- [samplers.jl](samplers.jl): coupling functions required by the simulation.
- [meeting_times_max-reflection_random_walk.jl](meeting_times_max-reflection_random_walk.jl): meeting-time simulation.
- [figures/make_rw_meeting_times_fig.jl](figures/make_rw_meeting_times_fig.jl): plots the saved meeting-time data.

The source repository is unchanged. This folder preserves the original relative paths and code; no algorithm or parameter changes were made.

From the repository root, install the Julia packages and run the simulation:

```sh
cd sphere_random_walk
julia -e 'using Pkg; Pkg.add(["ProgressBars", "JLD", "Plots", "LaTeXStrings"])'
mkdir -p data
julia --threads=auto meeting_times_max-reflection_random_walk.jl
```

Then run the plotting script from its own folder:

```sh
cd figures
julia make_rw_meeting_times_fig.jl
```

The simulation writes `data/mr_random_walk_meeting_times.jld`. The plotting script reads that file and writes three PNG figures in `figures/`.

The original settings are retained: 1,000,000 repetitions for each of 12 dimensions and 101 step sizes. This is a large simulation; it was not rerun as part of copying the files. Generated data and figures are not included.
