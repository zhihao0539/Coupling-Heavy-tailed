#!/usr/bin/env python3
"""Append 900 independent bright-side initial pairs to the original 100."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
original=np.load(ROOT/'initial_states_100.npz')
rng=np.random.default_rng(26092910)
samples=[]
remaining=1800
while remaining:
    g=rng.normal(size=(max(2048,remaining*2),101))
    g/=np.linalg.norm(g,axis=1,keepdims=True)
    accepted=g[g[:,-1]<.1][:remaining]
    samples.append(accepted)
    remaining-=len(accepted)
znew=np.concatenate(samples).reshape(900,2,101)
xnew=znew[:,:,:100]/(.1-znew[:,:,-1:])
np.savez_compressed(ROOT/'initial_states.npz',
                    initial_states=np.concatenate([original['initial_states'],xnew]),
                    sub_cauchy_sphere_states=np.concatenate([original['sub_cauchy_sphere_states'],znew]))
