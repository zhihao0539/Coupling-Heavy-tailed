using Plots
using LaTeXStrings
using JLD


###
# This script assumes that the script meeting_times_max-reflection_random_walk.jl has already been run
###
N = load("../data/mr_random_walk_meeting_times.jld", "N")
ds = load("../data/mr_random_walk_meeting_times.jld", "ds")
meeting_times = load("../data/mr_random_walk_meeting_times.jld", "meeting_times")
std_err = sqrt.(load("../data/mr_random_walk_meeting_times.jld", "var_meeting_times") ./ N)
h_scale = LinRange(-1.5, -0.5, load("../data/mr_random_walk_meeting_times.jld", "num_hs"))
hs = exp.(log.(ds) .* h_scale')


#Heat map of the meeting times as a function of the time step and dimension
heatmap(ds, h_scale, meeting_times', xscale = :log10, colorbar_scale = :log10)
xlabel!(LaTeXString("Dimension, \$ d\$"))
ylabel!(LaTeXString("Step size, \$ \\log(h)/\\log(d+1)\$"))
plot!(title = "Meeting Time of MR Coupling of Stereo RW")
plot!(colorbar_title = "Expected meeting time")
savefig("rw_expected_meeting_times_heatmap.png")


#Plot of meeting times as a function of dimension
plot()
for (i, h) = enumerate(h_scale)
	if h in [-1.5, -1.25, -1.0, -0.75, -0.5]
		plot!(ds, meeting_times[:, i], ribbon = 2. .* std_err[:, i], marker = :circle, ms = 2, msw = 0.5, label = "h = (d+1)^($(h))")
	end
end
plot!(xscale = :log10, yscale = :log10)
plot!(xlabel = LaTeXString("Dimension, \$ d\$"), ylabel = "Expected meeting time", legend_title = "Time step")
plot!(title = "Meeting time of MR coupling of stereo RW")
savefig("rw_expected_meeting_times.png")


plot()
for (i, d) in enumerate(ds)
	scatter!(hs[i, :] .* sqrt(d), meeting_times[i, :], label = "d = $(d)", ms = 4, msw = 0.5)
end
plot!(scale = :log10, size = (720, 480))
plot!(xlabel = LaTeXString("step size scaled by dimension \$d^{1/2}h\$"),ylabel = "Average meeting time", legend_title = "Dimension")
plot!(title = "Scaling of Meeting time of MR coupling of stereo RW")
plot!(labelfontsize = 12, titlefontsize = 15)
savefig("scaling_of_rw_expected_meeting_times.png")
