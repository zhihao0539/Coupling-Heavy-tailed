#!/usr/bin/env python3
"""Validate and summarize exactly 1,000 paired replicates of every method."""
from pathlib import Path
import csv,hashlib,json,platform
import numpy as np
ROOT=Path(__file__).resolve().parent
METHODS=['stereographic','euclidean','scs','dcs_bw']
FIELDS=['method','replicate','met','meeting_time','iterations','acceptance_rate_chain1','acceptance_rate_chain2','proposal_meetings','boundary_rejections','elapsed_seconds','provenance','seed']

def read(path):return list(csv.DictReader(path.open()))

def main():
    retained=read(ROOT/'initial_meeting_times_100.csv')
    rows=retained.copy()
    for method in ['scs','dcs_bw','euclidean']:
        for path in sorted((ROOT/'checkpoints').glob(f'{method}_*.csv')):rows.extend(read(path))
    native=read(ROOT/'stereographic_native_extension.csv')
    assert len(native)==900,'Wait until all 900 new stereographic replicates have finished.'
    # Use the native result for EVERY new stereographic replicate. No selection
    # between native/JAX streams by outcome, runtime, or meeting/censoring status.
    rows.extend(native)
    rows.sort(key=lambda r:(METHODS.index(r['method']),int(r['replicate'])))
    assert len(rows)==4000
    for method in METHODS:
        a=[r for r in rows if r['method']==method]
        assert sorted(int(r['replicate']) for r in a)==list(range(1000))
        for r in a:
            met=r['met'].lower()=='true';tau=int(r['meeting_time']);it=int(r['iterations'])
            assert (met and 1<=tau==it<=1000000) or (not met and tau==-1 and it==1000000)
            assert 0<=float(r['acceptance_rate_chain1'])<=1
            assert 0<=float(r['acceptance_rate_chain2'])<=1
    for r in retained:
        match=next(v for v in rows if v['method']==r['method'] and v['replicate']==r['replicate'])
        assert match==r
    with (ROOT/'meeting_times.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(26092916)
    bootstrap_indices=rng.integers(0,1000,size=(10000,1000),dtype=np.int32)
    summary=[];all_tau={}
    for method in METHODS:
        a=[r for r in rows if r['method']==method]
        tau=np.array([int(r['meeting_time']) if r['met'].lower()=='true' else np.inf for r in a])
        stopped=np.minimum(tau,1e6);finite=tau[np.isfinite(tau)];sorted_tau=np.sort(tau)
        boot=stopped[bootstrap_indices].mean(axis=1);ci=np.quantile(boot,[.025,.975])
        acc=np.array([(.5*(float(r['acceptance_rate_chain1'])+float(r['acceptance_rate_chain2']))) for r in a])
        it=np.array([int(r['iterations']) for r in a])
        summary.append({'method':method,'replicates':1000,'met':int(np.isfinite(tau).sum()),'censored':int(np.isinf(tau).sum()),
                        'km_median':float(sorted_tau[499]) if np.isfinite(sorted_tau[499]) else None,
                        'finite_only_median':float(np.median(finite)),
                        'restricted_mean_1m':float(stopped.mean()),'mean_95_low':float(ci[0]),'mean_95_high':float(ci[1]),
                        'q90':float(sorted_tau[899]) if np.isfinite(sorted_tau[899]) else None,
                        'finite_max':int(finite.max()),'mean_stopped_acceptance':float(acc.mean()),
                        'pooled_stopped_acceptance':float(np.dot(acc,it)/it.sum())})
        all_tau[method]=tau
    ratio=all_tau['dcs_bw'].mean()/all_tau['scs'].mean()
    boot_ratio=all_tau['dcs_bw'][bootstrap_indices].mean(axis=1)/all_tau['scs'][bootstrap_indices].mean(axis=1)
    result={'methods':summary,'dcs_to_scs_mean_ratio':float(ratio),'paired_bootstrap_ratio_95':np.quantile(boot_ratio,[.025,.975]).tolist(),
            'bootstrap_seed':26092916,'bootstrap_resamples':10000,
            'mean_definition':'sample mean of min(tau, 1e6); unrestricted sample mean when no censoring',
            'median_definition':'first empirical survival crossing of 0.5, retaining censoring',
            'censoring_rule':'no exact meeting by 1,000,000 iterations',
            'acceptance_legend_definition':'unweighted average of the two chains and then of replicate-specific acceptance rates up to meeting/censoring',
            'data_provenance':'100 retained + 900 new per method; all new stereographic rows use the native backend'}
    (ROOT/'results_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (ROOT/'results_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    init=np.load(ROOT/'initial_states.npz');prefix=np.load(ROOT/'initial_states_100.npz')
    assert init['initial_states'].shape==(1000,2,100)
    assert init['sub_cauchy_sphere_states'].shape==(1000,2,101)
    assert np.array_equal(init['initial_states'][:100],prefix['initial_states'])
    assert np.array_equal(init['sub_cauchy_sphere_states'][:100],prefix['sub_cauchy_sphere_states'])
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text())
    manifest['new_stereographic_backend']={'implementation':'stereographic_native.cpp','rng':'std::mt19937_64','seed_rule':'splitmix64(26092933 + global_replicate_id)','replicates':'100..999','all_new_stereographic_rows_use_this_backend':True,'validation':'native_validation.json'}
    manifest['all_4000_observations_complete']=True
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    report={'all_checks_passed':True,'observations':len(rows),'unique_pairs_per_method':1000,'retained_observations_verified':400,
            'shared_initial_arrays_verified':True,'native_only_new_stereographic_selection':True,
            'numpy':np.__version__,'python':platform.python_version(),
            'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.cpp','.csv','.npz','.npy','.json','.md','.pdf','.png','.txt'} and p.name not in {'progress.json','native_progress.json','native_process.json','final_validation.json'}}}
    (ROOT/'final_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
