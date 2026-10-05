#!/usr/bin/env python3
"""Summarize the saved trials and reproduce the supplied 14 x 5 inch layout."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
METHODS={'stereographic':'Stereographic','euclidean':'Euclidean'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT)
    args=parser.parse_args();root=args.input
    config=json.loads((root/'configuration.json').read_text())
    with (root/'meeting_times.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2*config['repetitions']
    assert len({(r['method'],r['replicate']) for r in rows})==len(rows)
    tau={};summary=[];byrep={}
    for method,label in METHODS.items():
        rr=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['replicate']))
        assert [int(r['replicate']) for r in rr]==list(range(config['repetitions']))
        assert all(int(r['iterations'])==config['iterations'] for r in rr)
        assert all(int(r['nonfinite_proposal_targets'])==0 for r in rr)
        assert all(r['meeting_time']==r['exact_meeting_time'] for r in rr)
        raw=np.array([int(r['meeting_time']) for r in rr]);t=np.where(raw<0,np.inf,raw)
        assert np.all((raw==-1)|((raw>=1)&(raw<=config['iterations'])))
        tau[method]=t;byrep[method]=rr
        finite=t[np.isfinite(t)];restricted=np.minimum(t,config['iterations'])
        ac=np.array([[float(r['acceptance_rate_chain1']),float(r['acceptance_rate_chain2'])] for r in rr]).mean(1)
        row=dict(method=method,repetitions=len(rr),met=len(finite),censored=int(np.isinf(t).sum()),
            mean_meeting_time=float(t.mean()) if len(finite)==len(t) else None,
            restricted_mean=float(restricted.mean()),mean_mcse=float(restricted.std(ddof=1)/np.sqrt(len(t))),
            median=float(np.median(t)),q95=float(np.quantile(t,.95)),maximum_observed=float(finite.max()),
            acceptance_rate=float(ac.mean()),acceptance_mcse=float(ac.std(ddof=1)/np.sqrt(len(ac))))
        summary.append(row)
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fields=['meeting_time_sp','meeting_time_eu','sp_acc1','sp_acc2','eu_acc1','eu_acc2']
    with (root/'original_format_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i in range(config['repetitions']):
            sp,eu=byrep['stereographic'][i],byrep['euclidean'][i]
            writer.writerow(dict(zip(fields,[
                int(sp['meeting_index_zero_based']) if int(sp['meeting_time'])>0 else 'inf',
                int(eu['meeting_index_zero_based']) if int(eu['meeting_time'])>0 else 'inf',
                sp['acceptance_rate_chain1'],sp['acceptance_rate_chain2'],
                eu['acceptance_rate_chain1'],eu['acceptance_rate_chain2']])))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    finite_all=np.concatenate([v[np.isfinite(v)] for v in tau.values()])
    bins=np.linspace(finite_all.min(),finite_all.max(),51)
    iterations=np.arange(config['iterations']+1)
    for row,color in zip(summary,['tab:blue','tab:orange']):
        method=row['method'];values=tau[method];finite=values[np.isfinite(values)]
        axes[0].hist(finite,bins=bins,weights=np.full(len(finite),1/len(finite)),alpha=.7,
                     edgecolor='black',linewidth=.5,color=color,
                     label=f"{METHODS[method]}, acc={row['acceptance_rate']:.3f}")
        survival=1-np.searchsorted(np.sort(values),iterations,side='right')/len(values)
        axes[1].plot(iterations,survival,linewidth=1.5,color=color,label=METHODS[method])
    axes[0].set(xlabel='Meeting time',ylabel='Probability',title='Distribution of meeting times')
    axes[1].set(xlabel='Iteration $t$',ylabel=r'$\widehat{\mathbb{P}}(\tau > t)$',title='Meeting-time survival')
    for ax in axes:
        ax.legend();ax.grid(True,alpha=.3);ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(root/'student_t_meeting_times.pdf',bbox_inches='tight')
    fig.savefig(root/'student_t_meeting_times.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
