#!/usr/bin/env python3
"""Extend the saved 100 paired replicates to 1,000 using the validated JAX kernels.

python run_1000_replicates.py --workers 4 --methods scs dcs_bw euclidean
Then run run_stereographic_native.py and finalize_results.py for the reported
four-method comparison. The optional JAX stereographic backend is not used
for any of the 900 new stereographic observations in the delivered figure.
Each independent batch is checkpointed; rerunning resumes completed batches.
The first 100 observations per method are retained, and IDs 100..999 are new.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, json, multiprocessing, os, time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent.parent/'work'/'mpl'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
FIELDS=['method','replicate','met','meeting_time','iterations','acceptance_rate_chain1','acceptance_rate_chain2','proposal_meetings','boundary_rejections','elapsed_seconds','provenance','seed']
METHODS=['scs','dcs_bw','euclidean','stereographic']
_cache={}

def execute_batch(job):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import coupling_experiment as core
    method,start,stop,batch_size=job
    checkpoint=ROOT/'checkpoints'/f'{method}_{start:04d}.csv'
    if checkpoint.exists():
        rows=list(csv.DictReader(checkpoint.open()))
        if len(rows)==stop-start:return str(checkpoint),0.
    if 'initial' not in _cache:
        _cache['initial']=np.load(ROOT/'initial_states.npz')
        _cache['theta']=jnp.asarray(np.load(ROOT/'dcs_theta.npy'))
    if method not in _cache:
        x=jnp.asarray(_cache['initial']['initial_states'])
        if method=='scs':native=jnp.asarray(_cache['initial']['sub_cauchy_sphere_states'])
        elif method=='dcs_bw':native=core.inverse_ball(x,_cache['theta'])
        elif method=='euclidean':native=x
        else:native=core.stereo_inverse(x)
        h={'scs':.15,'dcs_bw':.14,'euclidean':.26,'stereographic':14.}[method]
        scalar=core.make_runner(method,h,_cache['theta'],1_000_000)
        batched=jax.jit(jax.vmap(scalar))
        seed={'scs':26092913,'dcs_bw':26092923,'euclidean':26092943,'stereographic':26092933}[method]
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(seed),i))(jnp.arange(batch_size))
        compiled=batched.lower(keys,native[:batch_size,0],native[:batch_size,1]).compile()
        _cache[method]=(native,compiled,seed)
    native,run,seed=_cache[method]
    ids=list(range(start,stop))
    # Padding only affects unreported dummy pairs; every real stream uses global ID.
    padded=ids+[ids[-1]]*(batch_size-len(ids))
    idx=jnp.asarray(padded,dtype=jnp.int32)
    keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(seed),i))(idx)
    begun=time.perf_counter()
    vals=run(keys,native[idx,0],native[idx,1])
    vals=[np.asarray(v) for v in vals[:6]]
    elapsed=time.perf_counter()-begun
    rows=[]
    for j,i in enumerate(ids):
        it,met,a,b,br,pm=[v[j].item() for v in vals]
        assert it>0 and (met or it==1_000_000)
        rows.append({'method':method,'replicate':i,'met':bool(met),'meeting_time':it if met else -1,'iterations':it,
                     'acceptance_rate_chain1':a/it,'acceptance_rate_chain2':b/it,'proposal_meetings':pm,
                     'boundary_rejections':br,'elapsed_seconds':elapsed/len(ids),'provenance':'new_900_extension','seed':seed})
    temporary=checkpoint.with_suffix('.tmp')
    with temporary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    temporary.replace(checkpoint)
    return str(checkpoint),elapsed

def merge():
    rows=list(csv.DictReader((ROOT/'initial_meeting_times_100.csv').open()))
    for path in sorted((ROOT/'checkpoints').glob('*.csv')):
        rows.extend(csv.DictReader(path.open()))
    ids=[(r['method'],int(r['replicate'])) for r in rows]
    assert len(set(ids))==len(ids)
    rows.sort(key=lambda r:(METHODS.index(r['method']),int(r['replicate'])))
    temporary=ROOT/'meeting_times.tmp'
    with temporary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    temporary.replace(ROOT/'meeting_times.csv')
    return {m:sum(r['method']==m for r in rows) for m in METHODS}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=['scs','dcs_bw','euclidean'])
    args=p.parse_args()
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    jobs=[]
    for method in args.methods:
        size={'scs':16,'dcs_bw':16,'euclidean':4,'stereographic':16}[method]
        for start in range(100,1000,size):jobs.append((method,start,min(start+size,1000),size))
    begun=time.time();counts=merge()
    print('Starting/resuming:',counts,flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(execute_batch,j):j for j in jobs}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future);path,elapsed=future.result()
                counts=merge()
                print(f'{job[0]} {job[1]}:{job[2]} complete in {elapsed:.1f}s; counts={counts}',flush=True)
            progress={'counts':counts,'elapsed_seconds':time.time()-begun,'pending_batches':len(pending),'complete':not pending}
            (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
    assert all(counts[m]==1000 for m in args.methods),counts
    print('COMPLETE:',counts,'elapsed',time.time()-begun,flush=True)

if __name__=='__main__':main()
