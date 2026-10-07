# Spherical random walk — Section 5.1

The following three Julia files come from [Coupling-on-Sphere](https://github.com/zhihao0539/Coupling-on-Sphere/tree/4e30ceb849872d95c44b8f96abafa6043a98ca56):

- [samplers.jl](samplers.jl): coupling functions required by the simulation.
- [meeting_times_max-reflection_random_walk.jl](meeting_times_max-reflection_random_walk.jl): meeting-time simulation.
- [make_rw_meeting_times_fig.jl](make_rw_meeting_times_fig.jl): plots the saved meeting-time data.

The source repository is unchanged. The plotting script has been moved into this folder, and its input paths now use `data/` instead of `../data/`. No algorithm or parameter changes were made.

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
