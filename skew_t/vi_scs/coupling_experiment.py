#!/usr/bin/env python3
"""Reproducible extension of draft Section 5.3: nu=2, d=100.

Dependencies: Python >=3.10, numpy, scipy, jax, matplotlib.
Shared target, DCS fitting, transformations and coupling kernels.
Use the experiment runners described in README.md to reproduce 1,000 pairs.
Input: initial_states.npz and saved fitted parameters alongside this module.
The axial SCS kernel is retained as a numerical validation reference for the
VI-fitted implementation; SCS proposals on the dark side are rejected directly.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, time
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent.parent/'work'/'mpl'))
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import betainc
from scipy import special, stats

OUT=Path(__file__).resolve().parent
D=100
DF=2.0
LAT=1.1
SCS_R=10/11
LIMIT=1_000_000
AXIS=np.r_[1/np.sqrt(2),-1/np.sqrt(2),np.zeros(D-2)]

def logtcdf(t,df=DF+D):
    w=jnp.clip(df/(df+t*t),0.,1.-jnp.finfo(jnp.float64).eps)
    b=betainc(df/2,0.5,w)
    return jnp.where(t<0,jnp.log(0.5)+jnp.log(jnp.maximum(b,jnp.finfo(jnp.float64).tiny)),jnp.log1p(-0.5*b))

def logpi(x):
    r2=jnp.sum(x*x,axis=-1)
    t=100*(x[...,0]-x[...,1])*jnp.sqrt((D+DF)/(DF+r2))
    return -(D+DF)/2*jnp.log1p(r2/DF)+logtcdf(t)

def mobius(s,delta):
    r2=jnp.sum(s*s,axis=-1,keepdims=True)
    dot=jnp.sum(s*delta,axis=-1,keepdims=True)
    a2=jnp.sum(delta*delta)
    den=1+2*dot+a2*r2
    return ((1-a2)*s+(1+2*dot+r2)*delta)/den

def unpack(theta):
    # All 201 parameters are fitted, as in the paper's Adam procedure.
    mu=theta[:D]; radius=jnp.exp(theta[D]); v=theta[D+1:]
    norm=jnp.sqrt(jnp.sum(v*v)+1e-24)
    delta=(jnp.tanh(norm)/norm)*v
    return mu,radius,delta

def ball_map_logj(s,theta):
    mu,radius,delta=unpack(theta)
    r2=jnp.sum(s*s,axis=-1)
    a2=jnp.sum(delta*delta)
    den=1+2*jnp.sum(s*delta,axis=-1)+a2*r2
    y=mobius(s,delta)
    # This identity avoids subtractive cancellation near the unit-ball boundary.
    gap=(1-a2)*(1-r2)/den
    x=mu+radius*y/jnp.sqrt(gap[...,None])
    logj=D*jnp.log(radius)+D*jnp.log1p(-a2)-D*jnp.log(den)-(1+D/2)*jnp.log(gap)
    return x,logj

def ball_logpi(s,theta):
    x,logj=ball_map_logj(s,theta)
    return logpi(x)+logj

def inverse_ball(x,theta):
    mu,radius,delta=unpack(theta)
    u=(x-mu)/radius
    y=u/jnp.sqrt(1+jnp.sum(u*u,axis=-1,keepdims=True))
    return mobius(y,-delta)

def scs_map(z):
    return SCS_R*LAT*z[...,:-1]/(LAT-1-z[...,-1:])

def scs_inverse(x):
    u=x/SCS_R
    a=jnp.sum(u*u,axis=-1)+LAT**2
    b=-2*LAT*(LAT-1); c=LAT**2-2*LAT
    m=(-b+jnp.sqrt(b*b-4*a*c))/(2*a)
    return jnp.concatenate([m[...,None]*u,((1-m)*LAT-1)[...,None]],axis=-1)

def scs_logpi(z):
    t=z[...,-1]; b=LAT-1
    # Surface measure Jacobian, constants retained for validation.
    jac=D*jnp.log(SCS_R*LAT)+jnp.log1p(-b*t)-(D+1)*jnp.log(b-t)
    return logpi(scs_map(z))+jac

def stereo_inverse(x):
    n2=jnp.sum(x*x,axis=-1,keepdims=True)
    return jnp.concatenate([20*x/(n2+100),(n2-100)/(n2+100)],axis=-1)

def stereo_logpi(z):
    x=10*z[...,:-1]/(1-z[...,-1:])
    return logpi(x)-D*jnp.log1p(-z[...,-1])

def uniform_ball(key,n):
    k1,k2=jax.random.split(key)
    g=jax.random.normal(k1,(n,D),dtype=jnp.float64)
    return g/jnp.linalg.norm(g,axis=-1,keepdims=True)*jax.random.uniform(k2,(n,1),dtype=jnp.float64)**(1/D)

@jax.jit
def vi_loss(theta,s):
    x,lj=ball_map_logj(s,theta)
    return -jnp.mean(logpi(x)+lj)

def fit_vi(seed=26092901,steps=4000,batch=2048):
    theta0=jnp.zeros(2*D+1).at[D].set(math.log(math.sqrt(DF)))
    @jax.jit
    def fit(key):
        def body(carry,k):
            theta,m,v,key=carry
            key,sub=jax.random.split(key)
            s=uniform_ball(sub,batch)
            loss,grad=jax.value_and_grad(vi_loss)(theta,s)
            m=.9*m+.1*grad; v=.999*v+.001*grad*grad
            # Decay reduces Monte Carlo noise without changing the objective.
            rate=.01*jnp.where(k<3000,1.,.2)
            update=(m/(1-.9**(k+1)))/(jnp.sqrt(v/(1-.999**(k+1)))+1e-8)
            theta=theta-rate*update
            return (theta,m,v,key),(loss,theta)
        last,hist=jax.lax.scan(body,(theta0,jnp.zeros_like(theta0),jnp.zeros_like(theta0),key),jnp.arange(steps))
        return last[0],hist
    start=time.perf_counter()
    theta,(loss,history)=fit(jax.random.key(seed))
    theta=np.asarray(theta); loss=np.asarray(loss); history=np.asarray(history)
    holdout=uniform_ball(jax.random.key(seed+500),20000)
    candidates=[theta,history[-500:].mean(axis=0)]
    vals=[float(vi_loss(jnp.asarray(t),holdout)) for t in candidates]
    theta=candidates[int(np.argmin(vals))]
    np.save(OUT/'dcs_theta.npy',theta)
    np.savez_compressed(OUT/'vi_history.npz',loss=loss,theta=history)
    mu,r,delta=unpack(jnp.asarray(theta))
    info={'beta':2.,'df':DF,'dimension':D,'seed':seed,'steps':steps,'batch_size':batch,
          'learning_rate':'.01 for 3000 steps, .002 thereafter','adam_beta1':.9,'adam_beta2':.999,
          'initial_radius':math.sqrt(DF),'initial_mu':'zero','initial_delta':'zero',
          'chosen_candidate':['last','last_500_parameter_average'][int(np.argmin(vals))],
          'holdout_losses':vals,'initial_holdout_loss':float(vi_loss(theta0,holdout)),
          'radius':float(r),'mu':np.asarray(mu).tolist(),'delta':np.asarray(delta).tolist(),
          'mu_norm':float(jnp.linalg.norm(mu)),'delta_norm':float(jnp.linalg.norm(delta)),
          'mu_along_skewness':float(jnp.dot(mu,AXIS)),
          'delta_along_skewness':float(jnp.dot(delta,AXIS)),
          'elapsed_seconds':time.perf_counter()-start,
          'note':'Independent refit of the paper\'s reverse-KL Mobius procedure on the draft\'s nu=2 skew-t. Original fitted numbers and ball-walk radius were not published.'}
    (OUT/'dcs_parameters.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k not in ('mu','delta')},indent=2),flush=True)

def exact_target(rng,n):
    """Exact AC skew-t via a skew-normal/chi-square mixture."""
    direction=AXIS
    skew=math.sqrt(20000/20001)
    g=rng.normal(size=(n,D))
    gparallel=g@direction
    z=g-gparallel[:,None]*direction
    z+=(skew*np.abs(rng.normal(size=n))+math.sqrt(1-skew**2)*gparallel)[:,None]*direction
    return z/np.sqrt(rng.chisquare(DF,size=(n,1))/DF)

def tune():
    # Independent stationary draws avoid confusing stopped-path acceptance with
    # the acceptance rate of a stationary marginal sampler.
    rng=np.random.default_rng(26092902)
    x=exact_target(rng,40000)
    theta=jnp.asarray(np.load(OUT/'dcs_theta.npy'))
    rows=[]
    for method in ['scs','dcs_bw']:
        z=np.asarray(scs_inverse(x) if method=='scs' else inverse_ball(x,theta))
        logfun=jax.jit(scs_logpi) if method=='scs' else jax.jit(lambda s:ball_logpi(s,theta))
        lp=np.asarray(logfun(jnp.asarray(z)))
        noise=rng.normal(size=z.shape)
        if method=='scs':
            noise-=np.sum(noise*z,axis=1,keepdims=True)*z
            grid=[.03,.05,.08,.1,.15,.2,.3,.5,.75,1.,1.5,2.,3.,5.,10.,14.]
        else:
            noise/=np.linalg.norm(noise,axis=1,keepdims=True)
            noise*=rng.uniform(size=(len(z),1))**(1/D)
            grid=[.04,.06,.08,.10,.12,.14,.16,.18,.20,.22,.24,.26,.28,.3,.35,.4]
        for h in grid:
            p=z+h*noise
            if method=='scs':
                p/=np.linalg.norm(p,axis=1,keepdims=True)
                valid=p[:,-1]<LAT-1
            else:
                valid=np.sum(p*p,axis=1)<1
            safe=np.where(valid[:,None],p,z)
            lpp=np.asarray(logfun(jnp.asarray(safe)))
            a=valid*np.exp(np.minimum(0,lpp-lp))
            esjd=a*np.sum((p-z)**2,axis=1)
            rows.append({'method':method,'step':h,'acceptance':float(a.mean()),
                         'acceptance_se':float(a.std(ddof=1)/math.sqrt(len(a))),
                         'boundary_rejection':float(1-valid.mean()),'native_esjd':float(esjd.mean())})
    with (OUT/'stationary_step_tuning.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Retune SCS to the draft's 40% acceptance goal after changing its boundary
    # rule. Also retain h=3 as a separate sensitivity run.
    # DCS has no published step; select near 40% acceptance using an independent
    # pilot, then freeze it for all meeting-time replicates.
    selected=min([r for r in rows if r['method']=='dcs_bw'],key=lambda r:abs(r['acceptance']-.40))
    scs=min([r for r in rows if r['method']=='scs'],key=lambda r:abs(r['acceptance']-.40))
    result={'dcs_bw':selected,'scs':scs,'seed':26092902,'stationary_draws':len(x),
            'scs_rule':'Choose h nearest 0.40 marginal acceptance; retain old h=3 in a sensitivity run.',
            'dcs_rule':'Choose gamma/R nearest 0.40 marginal acceptance on independent exact stationary draws.'}
    (OUT/'step_selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def spherical_logq(z,p,h):
    c=jnp.dot(z,p)
    safe=jnp.maximum(c,1e-150)
    value=-(D+1)*jnp.log(safe)-(1/(safe*safe)-1)/(2*h*h)
    return jnp.where(c>0,value,-jnp.inf)

def make_runner(method,h,theta=None,max_steps=LIMIT):
    if method=='scs': logfun=scs_logpi
    elif method=='stereographic': logfun=stereo_logpi
    elif method=='euclidean': logfun=logpi
    else: logfun=lambda s:ball_logpi(s,theta)
    sphere=method in ('scs','stereographic')
    @jax.jit
    def run(key,z1,z2):
        initial=(jnp.int32(0),z1,z2,logfun(z1),logfun(z2),key,jnp.array(False),jnp.int32(0),jnp.int32(0),jnp.int32(0),jnp.int32(0))
        def condition(s): return (s[0]<max_steps)&(~s[6])
        def body(s):
            t,a,b,la,lb,key,met,ac1,ac2,br,pm=s
            key,kn,km,ku,kr=jax.random.split(key,5)
            g=jax.random.normal(kn,a.shape,dtype=jnp.float64)
            if sphere:
                v=h*(g-a*jnp.dot(a,g))
                p=(a+v)/jnp.linalg.norm(a+v)
                ratio=spherical_logq(b,p,h)-spherical_logq(a,p,h)
                common=jnp.log(jax.random.uniform(km,dtype=jnp.float64))<=jnp.minimum(0.,ratio)
                # A Householder reflection maps a to b and the tangent spaces;
                # equivalent to the draft's reflection followed by transport.
                e=(a-b)/jnp.maximum(jnp.linalg.norm(a-b),1e-150)
                reflected=p-2*e*jnp.dot(e,p)
                reflected=reflected/jnp.linalg.norm(reflected)
                q=jnp.where(common,p,reflected)
            else:
                if method=='dcs_bw':
                    v=h*g/jnp.linalg.norm(g)*jax.random.uniform(kr,dtype=jnp.float64)**(1/D)
                else: v=h*g
                p=a+v
                if method=='dcs_bw': common=jnp.sum((p-b)**2)<=h*h
                else:
                    lr=(jnp.sum(v*v)-jnp.sum((p-b)**2))/(2*h*h)
                    common=jnp.log(jax.random.uniform(km,dtype=jnp.float64))<=jnp.minimum(0.,lr)
                e=(a-b)/jnp.maximum(jnp.linalg.norm(a-b),1e-150)
                q=jnp.where(common,p,b+v-2*e*jnp.dot(e,v))
            if method=='scs': ok1=p[-1]<LAT-1; ok2=q[-1]<LAT-1
            elif method=='dcs_bw': ok1=jnp.dot(p,p)<1; ok2=jnp.dot(q,q)<1
            else: ok1=jnp.array(True); ok2=jnp.array(True)
            # Each chain rejects its own out-of-domain proposal. No stepping out.
            psafe=jnp.where(ok1,p,a); qsafe=jnp.where(ok2,q,b)
            lp=logfun(psafe); lq=logfun(qsafe)
            u=jnp.log(jax.random.uniform(ku,dtype=jnp.float64))
            accept1=ok1&(u<=jnp.minimum(0.,lp-la))
            accept2=ok2&(u<=jnp.minimum(0.,lq-lb))
            a=jnp.where(accept1,psafe,a); b=jnp.where(accept2,qsafe,b)
            la=jnp.where(accept1,lp,la);lb=jnp.where(accept2,lq,lb)
            return t+1,a,b,la,lb,key,jnp.all(a==b),ac1+accept1.astype(jnp.int32),ac2+accept2.astype(jnp.int32),br+(~ok1).astype(jnp.int32)+(~ok2).astype(jnp.int32),pm+common.astype(jnp.int32)
        s=jax.lax.while_loop(condition,body,initial)
        return s[0],s[6],s[7],s[8],s[9],s[10],s[1],s[2]
    return run

def simulate(methods=('scs','dcs_bw'),n=100,max_steps=LIMIT,filename='new_meeting_times.csv',seed=26092903,steps_override=None):
    data=np.load(OUT/'initial_states.npz')
    x=data['initial_states'][:n]
    theta=jnp.asarray(np.load(OUT/'dcs_theta.npy'))
    tuning=json.loads((OUT/'step_selection.json').read_text())
    rows=[]
    for method in methods:
        h={'scs':tuning['scs']['step'],'dcs_bw':tuning['dcs_bw']['step'],'stereographic':14.,'euclidean':.26}[method]
        if steps_override and method in steps_override: h=steps_override[method]
        if method=='scs': native=data['sub_cauchy_sphere_states'][:n]
        elif method=='dcs_bw': native=np.asarray(inverse_ball(jnp.asarray(x),theta))
        elif method=='stereographic': native=np.asarray(stereo_inverse(jnp.asarray(x)))
        else: native=x
        run=make_runner(method,h,theta,max_steps)
        run.lower(jax.random.key(0),jnp.asarray(native[0,0]),jnp.asarray(native[0,1])).compile()
        for i in range(n):
            key=jax.random.fold_in(jax.random.key(seed+{'scs':10,'dcs_bw':20,'stereographic':30,'euclidean':40}[method]),i)
            start=time.perf_counter()
            vals=run(key,jnp.asarray(native[i,0]),jnp.asarray(native[i,1]))
            it,met,a,b,br,pm=[np.asarray(v).item() for v in vals[:6]]
            elapsed=time.perf_counter()-start
            rows.append({'method':method,'replicate':i,'met':bool(met),'meeting_time':it if met else -1,
                         'iterations':it,'acceptance_rate_chain1':a/it,'acceptance_rate_chain2':b/it,
                         'proposal_meetings':pm,'boundary_rejections':br,'elapsed_seconds':elapsed,
                         'provenance':'new_run','seed':seed+{'scs':10,'dcs_bw':20,'stereographic':30,'euclidean':40}[method]})
            with (OUT/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            if (i+1)%10==0: print(method,i+1,'/',n,'latest tau',it,'met',met,flush=True)
        sample=[r['meeting_time'] for r in rows if r['method']==method and r['met']]
        print(method,'mean',np.mean(sample),'median',np.median(sample),'max',np.max(sample),flush=True)
    return rows

def validate():
    rng=np.random.default_rng(26092904)
    x=exact_target(rng,2000)
    theta=jnp.asarray(np.load(OUT/'dcs_theta.npy'))
    s=inverse_ball(jnp.asarray(x),theta)
    xr,_=ball_map_logj(s,theta)
    e1=float(np.max(np.linalg.norm(np.asarray(xr)-x,axis=1)/(1+np.linalg.norm(x,axis=1))))
    z=scs_inverse(jnp.asarray(x)); xs=scs_map(z)
    e2=float(np.max(np.linalg.norm(np.asarray(xs)-x,axis=1)/(1+np.linalg.norm(x,axis=1))))
    ts=np.r_[-np.geomspace(.0001,1400,100),0,np.geomspace(.0001,1400,100)]
    e3=float(np.max(np.abs(np.asarray(logtcdf(jnp.asarray(ts)))-stats.t.logcdf(ts,D+DF))))
    # Forward Jacobian: ambient derivative restricted to an orthonormal tangent basis.
    zz=np.asarray(z[0]); basis=np.linalg.svd(zz[None,:],full_matrices=True)[2][1:].T
    jac=np.asarray(jax.jacfwd(scs_map)(jnp.asarray(zz)))@basis
    numerical=np.linalg.slogdet(jac)[1]
    analytic=float(scs_logpi(zz)-logpi(scs_map(zz)))
    e4=abs(numerical-analytic)
    ss=np.asarray(s[0]);jacb=np.asarray(jax.jacfwd(lambda q:ball_map_logj(q,theta)[0])(ss))
    e5=abs(np.linalg.slogdet(jacb)[1]-float(ball_map_logj(ss,theta)[1]))
    # Check the geometric reflection equals tangent reflection + parallel transport.
    a,b=np.asarray(z[:2]);g=rng.normal(size=D+1);v=g-a*np.dot(a,g)
    c=np.dot(a,b);sin=np.sqrt(1-c*c)
    eab=(b-c*a)/sin;eba=(a-c*b)/sin
    vt=v-(a+b)*np.dot(b,v)/(1+c)+2*eba*np.dot(eab,v)
    house=(a-b)/np.linalg.norm(a-b)
    e6=float(np.max(np.abs(vt-(v-2*house*np.dot(house,v)))))
    result={'dcs_roundtrip_max_relative_error':e1,'scs_roundtrip_max_relative_error':e2,
            'student_t_logcdf_max_absolute_error_vs_scipy':e3,'scs_log_jacobian_error':e4,
            'dcs_log_jacobian_error':e5,'householder_vs_transport_max_error':e6}
    assert e1<1e-8 and e2<1e-8 and e3<1e-6 and e4<1e-8 and e5<1e-8 and e6<1e-10,result
    # Validate ball-walk maximal overlap and second marginal against known formulas.
    n=200000;h=.2;distance=.03
    g=rng.normal(size=(n,D));v=h*g/np.linalg.norm(g,axis=1,keepdims=True)*rng.random((n,1))**(1/D)
    a=np.zeros(D);b=np.zeros(D);b[0]=distance
    p=a+v;common=np.sum((p-b)**2,axis=1)<=h*h
    q=b+v.copy();q[:,0]=distance-v[:,0];q[common]=p[common]
    theoretical=special.betainc((D+1)/2,.5,1-(distance/(2*h))**2)
    result.update({'ball_overlap_empirical':float(common.mean()),'ball_overlap_exact':float(theoretical),
                   'second_marginal_max_coordinate_mean_error':float(np.max(abs((q-b).mean(axis=0)))),
                   'second_marginal_mean_squared_radius':float(np.mean(np.sum((q-b)**2,axis=1))),
                   'expected_squared_radius':h*h*D/(D+2)})
    assert abs(common.mean()-theoretical)<.005
    assert np.max(abs((q-b).mean(axis=0)))<.0003
    assert abs(result['second_marginal_mean_squared_radius']-result['expected_squared_radius'])<.00005
    result['all_passed']=True
    (OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fit','tune','run','validate','all'],default='all')
    args=p.parse_args()
    if args.stage in ('fit','all'):fit_vi()
    if args.stage in ('tune','all'):tune()
    if args.stage in ('validate','all'):validate()
    if args.stage in ('run','all'):simulate()
