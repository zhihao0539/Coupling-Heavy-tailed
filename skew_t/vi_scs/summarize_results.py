#!/usr/bin/env python3
"""Merge rerun VI-SCS with the three unchanged methods and summarize censoring."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
METHODS=['stereographic','euclidean','scs','dcs_bw']
def read(name):return list(csv.DictReader((ROOT/name).open()))
def times(rows):
    return np.array([int(r['meeting_time']) if r['met'].lower()=='true' else np.inf for r in rows])
def main():
    previous=read('previous_meeting_times.csv');new=read('scs_vi_meeting_times.csv')
    assert {r['method'] for r in previous}=={'stereographic','euclidean','dcs_bw'}
    retained=previous
    assert len(new)==1000 and len(retained)==3000
    rows=retained+new;rows.sort(key=lambda r:(METHODS.index(r['method']),int(r['replicate'])))
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        assert [int(r['replicate']) for r in selected]==list(range(1000))
        for r in selected:
            it=int(r['iterations']);met=r['met'].lower()=='true';tau=int(r['meeting_time'])
            assert (met and tau==it and 1<=it<=1000000) or (not met and tau==-1 and it==1000000)
            assert 0<=float(r['acceptance_rate_chain1'])<=1 and 0<=float(r['acceptance_rate_chain2'])<=1
    old_index={(r['method'],r['replicate']):r for r in retained}
    assert all(r==old_index[(r['method'],r['replicate'])] for r in rows if r['method']!='scs')
    with (ROOT/'meeting_times.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(previous[0]));w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(26093006);indices=rng.integers(0,1000,size=(10000,1000),dtype=np.int32)
    summary=[];all_times={}
    for method in METHODS:
        selected=[r for r in rows if r['method']==method];tau=times(selected);all_times[method]=tau
        stopped=np.minimum(tau,1e6);finite=tau[np.isfinite(tau)];sorted_tau=np.sort(tau)
        ci=np.quantile(stopped[indices].mean(axis=1),[.025,.975]);it=np.array([int(r['iterations']) for r in selected])
        acc=np.array([.5*(float(r['acceptance_rate_chain1'])+float(r['acceptance_rate_chain2'])) for r in selected])
        summary.append({'method':method,'replicates':1000,'met':len(finite),'censored':1000-len(finite),
                        'km_median':float(sorted_tau[499]) if np.isfinite(sorted_tau[499]) else None,
                        'finite_only_median':float(np.median(finite)),'restricted_mean_1m':float(stopped.mean()),
                        'mean_95_low':float(ci[0]),'mean_95_high':float(ci[1]),
                        'q90':float(sorted_tau[899]) if np.isfinite(sorted_tau[899]) else None,
                        'finite_max':int(finite.max()),'mean_stopped_acceptance':float(acc.mean()),
                        'pooled_stopped_acceptance':float(np.dot(acc,it)/it.sum())})
    new_tau=all_times['scs'];dcs_tau=all_times['dcs_bw']
    assert np.isfinite(new_tau).all() and np.isfinite(dcs_tau).all()
    newboot=new_tau[indices].mean(axis=1)
    result={'methods':summary,
            'dcs_to_scs_vi_mean_ratio':float(dcs_tau.mean()/new_tau.mean()),
            'dcs_to_scs_vi_ratio_95':np.quantile(dcs_tau[indices].mean(axis=1)/newboot,[.025,.975]).tolist(),
            'bootstrap_seed':26093006,'bootstrap_resamples':10000,
            'mean_definition':'mean of min(tau, 1e6); unrestricted sample mean if all met',
            'median_definition':'first empirical survival crossing at 0.5',
            'provenance':'1000 new VI-SCS rows; 3000 DCS, Euclidean and stereographic rows retained unchanged',
            'stationary_scs_acceptance':json.loads((ROOT/'scs_step_selection.json').read_text())}
    (ROOT/'results_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (ROOT/'results_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    report={'passed':True,'records':len(rows),'new_scs_records':len(new),'unchanged_other_method_records':len(retained),
            'initial_states_sha256':hashlib.sha256((ROOT/'initial_states.npz').read_bytes()).hexdigest(),
            'geometric_checks':json.loads((ROOT/'scs_validation.json').read_text())}
    (ROOT/'data_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
