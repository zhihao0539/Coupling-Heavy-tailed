using LinearAlgebra
using Random
using ProgressBars
using Statistics
using JLD

include("samplers.jl")

function sim_meeting_times(N, h, d)
	τs = Array{Float64, 1}(undef, N)

	Threads.@threads for i in axes(τs, 1)
		RNG = Xoshiro(i)
		x = normalize(randn(RNG, d+1))
		X_new = copy(x)
		y = -copy(x)
		G = randn(RNG, d+1)
		G̃ = randn(RNG, d+1)
		k = 0
		while x != y
			randn!(RNG, G)
			max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
			k += 1
		end
		τs[i] = k
	end

	return τs
end

N = 1_000_000
ds = [5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 500, 1000]
num_hs = 101
meeting_times = zeros(length(ds), num_hs)
var_meeting_times = zeros(length(ds), num_hs)

for (i, d) in enumerate(ds)
	progress_iterable = tqdm(enumerate(exp.(LinRange(-1.5, -0.5, num_hs) .* log(d+1))))
	set_postfix(progress_iterable, d = d)
	for (j, h) in progress_iterable
		τ = sim_meeting_times(N, h, d)
		meeting_times[i, j] = mean(τ)
		var_meeting_times[i, j] = var(τ)
	end
end

meeting_time_ste = sqrt.(var_meeting_times ./ N)

save("data/mr_random_walk_meeting_times.jld", "N", N, "ds", ds, "num_hs", num_hs, "meeting_times", meeting_times, "var_meeting_times", var_meeting_times)