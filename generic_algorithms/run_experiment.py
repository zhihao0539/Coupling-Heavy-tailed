"""Run repeated fixed-parameter couplings for any target log density."""

import pandas as pd
import numpy as np

from tqdm import tqdm
from multiprocess import Pool

from .algorithms import coupled_sampler



def experiment_iteration(rep, x, y, log_pdf, n_steps, stop_at_meeting, seed, settings, method, method_index, N_reps):
    """
    Wrapper function for one repetition of the meeting time experiment

    returns the row index of where to save result in results data frame and row of results

    """

    #fix random number generator
    if 'seed_offset' in settings:
        rng = np.random.default_rng(seed + rep + settings['seed_offset'])
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, method_index, rep]))
    
    #run repetition of meeting time experiment
    result = coupled_sampler(
        x, y, log_pdf, settings['h'], n_steps, rng, method=method,
        parameters=settings.get('parameters'),
        latitude=settings.get('latitude', 1.1),
        stop_at_meeting=stop_at_meeting,
    )

    #create dictionary of results of experiment repetition
    tau = result['meeting_time']
    out = dict(
        method=method, replicate=rep, met=bool(np.isfinite(tau)),
        meeting_time=int(tau) if np.isfinite(tau) else -1,
        iterations=result['iterations'],
        acceptance_rate_chain1=result['acceptance_rate_chain1'],
        acceptance_rate_chain2=result['acceptance_rate_chain2'],)

    #returns tuple: (row index, dictionary of results)
    return rep + method_index*N_reps, out

def run_experiment(log_pdf, initial_pairs, method_settings, n_steps, *, seed=42,
                   stop_at_meeting=False, output_csv=None):
    """Run each method on every initial pair, sequentially.

    ``initial_pairs`` has shape (repetitions, 2, dimension). Each entry in
    ``method_settings`` maps a method name to ``{'h': step_size,
    'parameters': fixed_map_parameters}``. ``latitude`` is optional for SCS,
    and ``seed_offset`` can preserve an application's previous random streams.
    Returns one result dictionary per method and repetition. Unmet meeting
    times are recorded as -1 in the returned rows and optional CSV.
    """
    pairs = np.asarray(initial_pairs, dtype=float)
    if pairs.ndim != 3 or pairs.shape[0] == 0 or pairs.shape[1] != 2 or pairs.shape[2] == 0:
        raise ValueError('initial_pairs must have shape (repetitions, 2, dimension).')
    if not np.isfinite(pairs).all():
        raise ValueError('Initial states must be finite.')
    if not method_settings:
        raise ValueError('Supply at least one method and its settings.')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')

    #create data frame to store experiment results
    N_reps = pairs.shape[0]
    result_df = pd.DataFrame(data = None, index=range(len(method_settings.items())*N_reps), 
        columns = ["method", "replicate", "met", "meeting_time", "iterations", "acceptance_rate_chain1", "acceptance_rate_chain2"])

    #Sequential run meeting time experiment for each method 
    for method_index, (method, settings) in enumerate(method_settings.items()):

        #define function run in parallel by imap
        pool_func = lambda rep : experiment_iteration(rep, pairs[rep, 0, :], pairs[rep, 1, :], log_pdf, n_steps, stop_at_meeting, seed, settings, method, method_index, N_reps)

        print(f"Running {method} experiment\n", flush=True)
        # run repetitions of meeting time experiment for given method in parallel
        # lets multiprocess choose the number of processes
        with Pool() as pool:
            for (row_ind, out_dict) in tqdm(pool.imap_unordered(pool_func, range(N_reps)), total = N_reps):
                result_df.loc[row_ind] = out_dict

    #coerce data types of results data frame
    result_df = result_df.astype({"method": "object", "replicate" : "Int64", "met" : bool, "meeting_time" : "Int64", "iterations" : "Int64", "acceptance_rate_chain1" : "Float64", "acceptance_rate_chain2": "Float64"})

    if output_csv is not None:
        #write results dataframe as csv file
        result_df.to_csv(output_csv)

    return result_df
