# Methods paragraph

Following the variational tuning approaches of Grazzi et al. (2026, Section 3) and Brešar and Mijatović (2026, Section 2.3.1 and Appendix D), we fitted the transformation parameters of SCS and DCS before running the coupling experiments by minimizing the reverse Kullback–Leibler divergence KL(qθ || π), where qθ is the pushforward of the uniform distribution on the corresponding native domain. For SCS, we fixed the observer latitude at 1.1 and optimized its horizontal position, the location vector and the scalar scale; for DCS, we fixed β = 2 and optimized the location, scale and Möbius parameters. Each fit used 4,000 Adam iterations with 2,048 Monte Carlo samples per iteration. We then tuned the proposal sizes separately to approximately 40% stationary acceptance and held all parameters fixed during the 1,000 coupling replicates, using the same Euclidean initial pairs for every method. SCS proposals landing on the dark side were rejected directly, without stepping out.

# Optional implementation detail

The learning rate was 0.01 for the first 3,000 iterations and 0.002 for the remaining 1,000 iterations. The final iterate and the average of the last 500 unconstrained iterates were compared using 20,000 independent samples from the base distribution; the average was selected for both samplers. Both fits optimized 201 parameters. No coupling meeting-time results were used to fit or select the transformation parameters. The DCS fit was retained from the preceding experiment, and SCS was fitted independently on the same ν = 2 target.

# Results paragraph

With variationally fitted transformations and proposal sizes tuned to approximately 40% stationary acceptance, both SCS and DCS-Ball walk met in all 1,000 replicates. Their mean meeting times were 40.88 and 1,691.96 iterations, respectively, giving a DCS-to-SCS mean ratio of 41.39 (paired bootstrap 95% interval: 38.92–43.87). The updated SCS scale was R = 1.32709 and its proposal step size was h = 0.08. Relative to the earlier SCS settings, whose mean meeting time was 16.20 iterations, variational fitting followed by acceptance-rate retuning increased the mean meeting time in this experiment. The variational objective measures the quality of the transformed-density approximation and does not directly optimize coupling meeting times.

# Figure caption

Meeting times of maximal-reflection couplings targeting the 100-dimensional skew-t distribution with ν = 2 and α = (100, −100, 0, …, 0), using 1,000 common Euclidean initial pairs. These pairs were generated using the original SCS transformation and were mapped into each algorithm's native state space using its own transformation. SCS and DCS transformations were fitted by reverse-KL variational inference before the meeting-time runs. Left: histograms conditional on meeting by 10⁶ iterations. Right: empirical meeting-time survival, including right-censored pairs. SCS rejects dark-side proposals directly. The proposal sizes are h = 0.08 for SCS and γ/R = 0.14 for DCS; their stationary acceptance estimates are 40.3% and 40.0%, respectively. The legend reports the unweighted mean of individual-chain acceptance fractions up to meeting or censoring. All SCS and DCS pairs met; 442 stereographic and 13 Euclidean pairs were censored at the cutoff.

# References

- Grazzi, S., Liu, S., Roberts, G. O., and Yang, J. (2026). *Sub-Cauchy Sampling: Escaping the Dark Side of the Moon*. Section 3, Equation (7). https://arxiv.org/pdf/2601.11066
- Brešar, M., and Mijatović, A. (2026). *Diffeomorphic Markov Chain Monte Carlo: fast mixing for heavy-tailed distributions*. Section 2.3.1 and Appendix D, Equation (32). https://arxiv.org/pdf/2608.04284

The shared optimization principle follows these references. The exact budget, learning-rate schedule, holdout selection and independent stationary acceptance tuning above describe this experiment; they are not claimed to reproduce every implementation choice or numerical setting in either reference paper.
