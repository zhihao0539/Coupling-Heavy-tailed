#!/usr/bin/env python3
"""Run 1,000 paired-initial-state trials, retain full-horizon acceptance."""
import argparse
import concurrent.futures as cf
import csv
import ctypes as ct
import hashlib
import json
import platform
import time
import numpy as np
from regression_coupling import ROOT,D,NU,R,STEPS,load_data,to_sphere,build_library,ptr,splitmix64

FIELDS = ['method','replicate','seed','met','meeting_time','meeting_index_zero_based','exact_meeting_time',
          'iterations','acceptance_rate_chain1','acceptance_rate_chain2','premeeting_acceptance_chain1',
          'premeeting_acceptance_chain2','proposal_meetings','nonfinite_proposal_targets','elapsed_seconds']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=1000)
    parser.add_argument('--iterations',type=int,default=20000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=str,default='.')
    args=parser.parse_args()
    if args.repetitions<=0 or args.iterations<=0 or args.workers<=0:
        parser.error('Counts must be positive.')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    X,y,scaling=load_data();lib=build_library()
    initial=np.random.RandomState(42).normal(0,10,(args.repetitions,2,D))
    np.savez(out/'initial_states.npz',initial_states=initial)
    (out/'standardization.json').write_text(json.dumps(scaling,indent=2)+'\n')
    config=dict(repetitions=args.repetitions,iterations=args.iterations,nu=NU,dimension=D,
        observations=len(y),intercept=False,radius=float(R),proposal_scales=STEPS,
        initial_distribution='independent N(0,100 I_14) pairs, shared across methods',initial_seed=42,
        initial_rng='NumPy RandomState MT19937; all pairs generated before sampling',
        transition_rng='std::mt19937_64, std::normal_distribution<double>',
        seed_bases={'stereographic':26100501,'euclidean':26100502},
        seed_rule='splitmix64(seed_base + 2*replicate)',tolerance=1e-12,
        acceptance_window='all iterations, including after meeting',meeting_time='one-based update count',
        data_sha256=hashlib.sha256((ROOT/'BostonHousing.csv').read_bytes()).hexdigest())
    cp=out/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:
        raise ValueError('Existing output uses different settings; choose another --output directory.')
    cp.write_text(json.dumps(config,indent=2)+'\n')
    path=out/'meeting_times.csv'
    rows=list(csv.DictReader(path.open())) if path.exists() else []
    seen={(r['method'],int(r['replicate'])) for r in rows}
    if len(seen)!=len(rows):raise ValueError('Duplicate saved result rows.')
    begun=time.perf_counter()
    def run(method,i):
        code=int(method=='stereographic');h=STEPS[method]
        a,b=(to_sphere(initial[i]) if code else initial[i]).copy()
        seed=splitmix64(config['seed_bases'][method]+2*i)
        result=np.empty(8,dtype=np.int64);start=time.perf_counter()
        lib.regression_run(ptr(X),ptr(y),len(y),NU,R,code,ptr(a),ptr(b),seed,args.iterations,h,1e-12,
                           result.ctypes.data_as(ct.POINTER(ct.c_int64)))
        tau,ac1,ac2,pre1,pre2,exact,pm,bad=result.tolist();denom=tau or args.iterations
        return dict(zip(FIELDS,[method,i,seed,bool(tau),tau if tau else -1,tau-1 if tau else -1,
            exact if exact else -1,args.iterations,ac1/args.iterations,ac2/args.iterations,
            pre1/denom,pre2/denom,pm,bad,time.perf_counter()-start]))
    jobs=[(method,i) for i in range(args.repetitions) for method in STEPS if (method,i) not in seen]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,*job) for job in jobs]
        for future in cf.as_completed(futures):
            rows.append(future.result());rows.sort(key=lambda r:(r['method'],int(r['replicate'])))
            temporary=path.with_suffix('.tmp')
            with temporary.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
            temporary.replace(path)
            if len(rows)%50==0 or len(rows)==2*args.repetitions:
                print(f'{len(rows)}/{2*args.repetitions} runs complete; {time.perf_counter()-begun:.1f}s',flush=True)
    runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                 workers=args.workers,wall_seconds_this_invocation=time.perf_counter()-begun)
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')

if __name__=='__main__':main()
