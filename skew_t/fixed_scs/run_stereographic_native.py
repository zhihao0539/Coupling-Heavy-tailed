#!/usr/bin/env python3
"""Run replicate IDs 100..999 of the validated stereographic kernel in C++.

The native source is mathematically equivalent to the JAX kernel. Its target,
Student-t CDF and deterministic single-step decisions were cross-checked.
Independent mt19937_64 streams use splitmix64(26092933 + replicate).
The original first 100 observations are supplied separately at final merge.
"""
from pathlib import Path
import argparse,concurrent.futures,csv,ctypes as ct,json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
FIELDS=['method','replicate','met','meeting_time','iterations','acceptance_rate_chain1','acceptance_rate_chain2','proposal_meetings','boundary_rejections','elapsed_seconds','provenance','seed']

def splitmix(x):
    mask=(1<<64)-1
    x=(x+0x9E3779B97F4A7C15)&mask
    x=((x^(x>>30))*0xBF58476D1CE4E5B9)&mask
    x=((x^(x>>27))*0x94D049BB133111EB)&mask
    return x^(x>>31)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    build=ROOT/'native_build';build.mkdir(exist_ok=True)
    library=build/('stereographic'+suffix)
    source=ROOT/'stereographic_native.cpp'
    command=['clang++' if sys.platform=='darwin' else 'g++','-O3','-std=c++17',
             '-dynamiclib' if sys.platform=='darwin' else '-shared', '-fPIC',str(source),'-o',str(library)]
    subprocess.run(command,check=True)
    lib=ct.CDLL(str(library));ptr=ct.POINTER(ct.c_double);iptr=ct.POINTER(ct.c_int64)
    lib.native_run.argtypes=[ptr,ptr,ct.c_uint64,ct.c_int,ct.c_double,iptr]
    x=np.load(ROOT/'initial_states.npz')['initial_states'];r2=np.sum(x*x,axis=-1,keepdims=True)
    z=np.ascontiguousarray(np.concatenate([20*x/(r2+100),(r2-100)/(r2+100)],axis=-1))
    def run(i):
        a=np.ascontiguousarray(z[i,0]);b=np.ascontiguousarray(z[i,1]);result=np.empty(5,dtype=np.int64)
        seed=splitmix(26092933+i);start=time.perf_counter()
        lib.native_run(a.ctypes.data_as(ptr),b.ctypes.data_as(ptr),seed,1000000,14.,result.ctypes.data_as(iptr))
        elapsed=time.perf_counter()-start;it,met,ac1,ac2,pm=result.tolist()
        assert it>0 and (met or it==1000000)
        return {'method':'stereographic','replicate':i,'met':bool(met),'meeting_time':it if met else -1,
                'iterations':it,'acceptance_rate_chain1':ac1/it,'acceptance_rate_chain2':ac2/it,
                'proposal_meetings':pm,'boundary_rejections':0,'elapsed_seconds':elapsed,
                'provenance':'new_900_native_extension','seed':seed}
    target=ROOT/'stereographic_native_extension.csv'
    rows=list(csv.DictReader(target.open())) if target.exists() else []
    seen={int(r['replicate']) for r in rows}
    begun=time.time()
    (ROOT/'native_process.json').write_text(json.dumps({'pid':os.getpid(),'start_time':begun,'workers':args.workers,'compile_command':command},indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run,i):i for i in range(100,1000) if i not in seen}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);rows.append(f.result())
            if done:
                rows.sort(key=lambda r:int(r['replicate']))
                temp=target.with_suffix('.tmp')
                with temp.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
                temp.replace(target)
            progress={'completed_new_replicates':len(rows),'total_new_replicates':900,'elapsed_seconds':time.time()-begun,'complete':not pending}
            (ROOT/'native_progress.json').write_text(json.dumps(progress,indent=2))
            if done and (len(rows)%25< len(done) or not pending):print(progress,flush=True)
    assert len(rows)==900
    print('Native stereographic extension complete.',flush=True)

if __name__=='__main__':main()
