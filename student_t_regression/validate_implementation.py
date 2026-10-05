#!/usr/bin/env python3
"""Independent density formulas, deterministic native/reference steps, coalescence."""
import ctypes as ct
import json
import numpy as np
from regression_coupling import *

def main():
    X,y,_=load_data();lib=build_library();rng=np.random.default_rng(20261005)
    density_error=0.;roundtrip_error=0.;step_error=0.;sphere_error=0.
    decisions=0
    for _ in range(200):
        theta=rng.normal(0,10,D)
        naive=-.5*(NU+1)*np.log1p((y-X@theta[:-1])**2/(NU*np.exp(2*theta[-1]))).sum()-len(y)*theta[-1]
        stable=log_density(theta,X,y)
        native=lib.regression_logtarget(ptr(X),ptr(y),len(y),NU,R,0,ptr(theta))
        density_error=max(density_error,abs(naive-stable),abs(native-stable))
        z=to_sphere(theta)
        roundtrip_error=max(roundtrip_error,float(np.max(np.abs(theta-from_sphere(z)))))
        for method in STEPS:
            code=int(method=='stereographic');dim=D+code
            a,b=rng.normal(0,10,(2,D))
            if rng.random()<.5:b=a+rng.normal(0,.005,D)
            if code:a,b=to_sphere(np.array([a,b]))
            g=rng.normal(size=dim);w,u=rng.random(2)
            ap,bp,fp=reference_step(a,b,g,w,u,method,STEPS[method],X,y)
            an=np.empty(dim);bn=np.empty(dim);fn=np.empty(4,dtype=np.int32)
            lib.regression_step(ptr(X),ptr(y),len(y),NU,R,code,ptr(a),ptr(b),ptr(g),w,u,STEPS[method],
                ptr(an),ptr(bn),fn.ctypes.data_as(ct.POINTER(ct.c_int)))
            assert np.array_equal(fn[:3],fp),(method,fn,fp)
            decisions+=3
            step_error=max(step_error,float(np.max(np.abs(ap-an))),float(np.max(np.abs(bp-bn))))
            if code:
                sphere_error=max(sphere_error,abs(np.linalg.norm(an)-1),abs(np.linalg.norm(bn)-1))
    assert density_error<1e-8
    assert roundtrip_error<1e-9
    assert step_error<1e-9
    assert sphere_error<1e-12
    for method in STEPS:
        a=rng.normal(size=D)
        if method=='stereographic':a=to_sphere(a)
        for _ in range(100):
            a,b,_=reference_step(a,a.copy(),rng.normal(size=len(a)),*rng.random(2),method,STEPS[method],X,y)
            assert np.array_equal(a,b)
    for u in [-1000.,1000.]:
        theta=np.zeros(D);theta[-1]=u
        native=lib.regression_logtarget(ptr(X),ptr(y),len(y),NU,R,0,ptr(theta))
        assert np.isfinite(native) and abs(native-log_density(theta,X,y))<1e-7
    result=dict(passed=True,max_density_absolute_error=density_error,max_projection_roundtrip_error=roundtrip_error,
        max_native_reference_step_error=step_error,max_sphere_norm_error=sphere_error,
        matching_acceptance_and_overlap_decisions=decisions,identical_chains_stay_identical=True,
        extreme_log_scale_checks_passed=True)
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
