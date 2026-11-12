"""UPAVE for d=50 and h(x)=x_1, with N(0,I_50) initial states."""
from pathlib import Path
import argparse
import csv
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0,str(FOLDER.parents[1]))

from generic_algorithms import load_parameters
from unbiased_poisson import pilot_lag
from unbiased_poisson.reservoir import upave_reservoir
from unbiased_poisson.kernels import (
    euclidean_kernels, stereographic_kernels, dcs_matrix_ball_kernels,
)
from target import RHO, make_log_density
from run_experiment import STEP_SIZES, RESULTS

D = 50
N_REP = 100
N_PILOT = 50
LAG_QUANTILE = .95
R = 10
MAX_ITERATIONS = 1_000_000
SEED = 20261112
PARAMETERS_FILE = RESULTS/f"d_{D:04d}"/"fitted_parameters.json"
OUTPUT = RESULTS/f"d_{D:04d}"/"asymptotic_variance"
LOG_DENSITY = make_log_density(D)


def save_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate_method(method,kernels,seed_offset,beta=None):
    def rinit(rng):
        return kernels['to_native'](rng.normal(size=D))
    def h(state):
        return float(kernels['to_euclidean'](state)[0])
    # Use the same original-space reference for every method, in native coordinates.
    y=kernels['to_native'](np.zeros(D))
    print(f'{method}: running {N_PILOT} independent lag-one pilots',flush=True)
    pilot=pilot_lag(kernels['single_kernel'],kernels['coupled_kernel'],rinit,
        n_pilot=N_PILOT,quantile=LAG_QUANTILE,seed=SEED+seed_offset+1_000_000,
        max_iterations=MAX_ITERATIONS)
    lag,k,ell=pilot['lag'],pilot['k'],pilot['ell']
    if max(lag+1,ell)>MAX_ITERATIONS:
        raise ValueError('Pilot settings exceed the cutoff; increase MAX_ITERATIONS.')
    save_csv([dict(pilot=i,meeting_time=int(t)) for i,t in enumerate(pilot['meeting_times'])],
        OUTPUT/f'{method}_pilot.csv')
    print(f'{method}: lag=k={lag}, ell={ell}, R={R}',flush=True)
    rows=[]
    seeds=np.random.SeedSequence(SEED+seed_offset).spawn(N_REP)
    for i,seed in enumerate(seeds):
        result=upave_reservoir(kernels['single_kernel'],kernels['coupled_kernel'],rinit,
            h,y,np.random.default_rng(seed),lag=lag,k=k,ell=ell,R=R,
            max_iterations=MAX_ITERATIONS)
        rows.append(dict(method=method,replicate=i,v=result['estimator'],varh=result['varh'],
            pi_h1=result['pi_h'][0],pi_h2=result['pi_h'][1],
            poisson_term1=result['poisson_terms'][0],poisson_term2=result['poisson_terms'][1],
            meeting_time1=int(result['meeting_times'][0]),meeting_time2=int(result['meeting_times'][1]),
            cost=result['cost'],cost_umcmc=result['cost_umcmc'],cost_poisson=result['cost_poisson'],
            elapsed_seconds=result['elapsed_seconds'],lag=lag,k=k,ell=ell,R=R,
            cutoff=MAX_ITERATIONS,step_size=STEP_SIZES[method][D],beta=beta,dimension=D,rho=RHO,
            initial_distribution='N(0,I)',observable='x1',seed=SEED+seed_offset))
        save_csv(rows,OUTPUT/f'{method}.csv')
        print(f'{method}: finished {i+1}/{N_REP}',flush=True)
    values=np.array([row['v'] for row in rows])
    mcse=float(values.std(ddof=1)/np.sqrt(N_REP))
    mean_cost=float(np.mean([row['cost'] for row in rows]))
    summary=dict(method=method,dimension=D,observable='x1',n_replicates=N_REP,
        mean_v=float(values.mean()),mcse=mcse,
        ci95_lower=float(values.mean()-1.96*mcse),ci95_upper=float(values.mean()+1.96*mcse),
        mean_varh=float(np.mean([row['varh'] for row in rows])),mean_cost=mean_cost,
        estimator_inefficiency=mean_cost*float(values.var(ddof=1)),
        lag=lag,k=k,ell=ell,R=R,cutoff=MAX_ITERATIONS,step_size=STEP_SIZES[method][D],beta=beta)
    save_csv([summary],OUTPUT/f'{method}_summary.csv')
    print(f"{method}: v(P,h)={summary['mean_v']:.6g}, MCSE={mcse:.6g}",flush=True)
    return summary


def run_stereographic():
    p=load_parameters(PARAMETERS_FILE,D,['stereographic'])['stereographic']
    kernels=stereographic_kernels(LOG_DENSITY,STEP_SIZES['stereographic'][D],**p)
    return estimate_method('stereographic',kernels,100_000)


def run_dcs():
    p=load_parameters(PARAMETERS_FILE,D,['dcs'])['dcs']
    kernels=dcs_matrix_ball_kernels(LOG_DENSITY,STEP_SIZES['dcs'][D],**p)
    return estimate_method('dcs',kernels,400_000,beta=p['beta'])


def run_euclidean():
    kernels=euclidean_kernels(LOG_DENSITY,STEP_SIZES['euclidean'][D])
    return estimate_method('euclidean',kernels,200_000)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('stereographic','dcs','euclidean'),
                        help='Run only one method; the default runs all three.')
    args=parser.parse_args(argv)
    runners=dict(stereographic=run_stereographic,dcs=run_dcs,euclidean=run_euclidean)
    for name,runner in runners.items():
        if args.method is None or args.method==name:
            runner()
    summaries=[]
    for method in runners:
        path=OUTPUT/f'{method}_summary.csv'
        if path.exists():
            with path.open(newline='') as handle:
                summaries.extend(csv.DictReader(handle))
    save_csv(summaries,OUTPUT/'summary.csv')


if __name__=='__main__':
    main()
