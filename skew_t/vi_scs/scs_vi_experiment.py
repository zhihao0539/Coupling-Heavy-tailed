#!/usr/bin/env python3
"""VI for SCS at fixed latitude, followed by rejection-only maximal coupling.

Follows SCS Section 3 (arXiv:2601.11066), using the same optimizer budget and
candidate rule as the saved DCS fit. All coordinates are centered unit-sphere
coordinates; the paper's sphere is translated upward by e_(d+1).
"""
from pathlib import Path
import argparse,csv,json,math,os,time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from scipy import special
import coupling_experiment as core

ROOT=Path(__file__).resolve().parent
D=100;LAT=1.1;B=LAT-1;LIMIT=1_000_000
OBSERVER_LIMIT=math.sqrt(1-B*B)
FIELDS=['method','replicate','met','meeting_time','iterations','acceptance_rate_chain1','acceptance_rate_chain2','proposal_meetings','boundary_rejections','elapsed_seconds','provenance','seed']

def initial_theta():
    return jnp.zeros(2*D+1).at[D].set(math.log(10/11))

def unpack(theta):
    mu=theta[:D];r=jnp.exp(theta[D]);v=theta[D+1:]
    norm=jnp.sqrt(jnp.sum(v*v)+1e-24)
    horizontal=OBSERVER_LIMIT*(jnp.tanh(norm)/norm)*v
    return mu,r,horizontal

def map_logj(z,theta):
    mu,r,horizontal=unpack(theta)
    t=z[...,-1];gap=B-t
    x=mu+r*(LAT*z[...,:-1]-(1+t[...,None])*horizontal)/gap[...,None]
    # Surface Jacobian: |det(DF restricted to an orthonormal tangent basis)|.
    cosine=1-B*t-jnp.sum(z[...,:-1]*horizontal,axis=-1)
    lj=D*jnp.log(r*LAT)+jnp.log(cosine)-(D+1)*jnp.log(gap)
    return x,lj

def logpi_sphere(z,theta):
    x,lj=map_logj(z,theta)
    return core.logpi(x)+lj

def inverse(x,theta):
    mu,r,horizontal=unpack(theta)
    a=(x-mu)/r-horizontal
    aa=jnp.sum(a*a,axis=-1)+LAT**2
    half_b=jnp.sum(a*horizontal,axis=-1)-LAT*B
    cc=jnp.sum(horizontal*horizontal)+B*B-1
    root=jnp.sqrt(half_b*half_b-aa*cc)
    # Stable positive root, including far-tail points and off-axis observers.
    m=jnp.where(half_b>0,-cc/(root+half_b),(-half_b+root)/aa)
    return jnp.concatenate([m[...,None]*a+horizontal,(B-LAT*m)[...,None]],axis=-1)

def uniform_cap(key,n):
    """IID uniform bright-side samples, exact rejection with no finite retry cap."""
    def cond(s):return s[2]<n
    def body(s):
        key,sub=jax.random.split(s[0])
        g=jax.random.normal(sub,(2*n,D+1),dtype=jnp.float64)
        z=g/jnp.linalg.norm(g,axis=1,keepdims=True)
        valid=z[:,-1]<B
        ids=jnp.nonzero(valid,size=n,fill_value=0)[0]
        return key,z[ids],jnp.sum(valid,dtype=jnp.int32)
    return jax.lax.while_loop(cond,body,(key,jnp.zeros((n,D+1)),jnp.int32(0)))[1]

@jax.jit
def loss(theta,z):return -jnp.mean(logpi_sphere(z,theta))

def fit():
    seed=26093001;steps=4000;batch=2048
    @jax.jit
    def chunk(carry,start):
        def body(carry,k):
            theta,m,v,key=carry
            key,sub=jax.random.split(key)
            z=uniform_cap(sub,batch)
            objective,grad=jax.value_and_grad(loss)(theta,z)
            m=.9*m+.1*grad;v=.999*v+.001*grad*grad
            rate=.01*jnp.where(k<3000,1.,.2)
            update=(m/(1-.9**(k+1)))/(jnp.sqrt(v/(1-.999**(k+1)))+1e-8)
            theta=theta-rate*update
            return (theta,m,v,key),(objective,theta)
        return jax.lax.scan(body,carry,start+jnp.arange(250))
    theta0=initial_theta();carry=(theta0,jnp.zeros_like(theta0),jnp.zeros_like(theta0),jax.random.key(seed))
    losses=[];history=[];begun=time.perf_counter()
    for start in range(0,steps,250):
        carry,(values,params)=chunk(carry,jnp.int32(start))
        values=np.asarray(values);params=np.asarray(params)
        assert np.isfinite(values).all() and np.isfinite(params).all()
        losses.append(values);history.append(params)
        print('VI steps',start+250,'mean loss',values.mean(),'elapsed',time.perf_counter()-begun,flush=True)
    losses=np.concatenate(losses);history=np.concatenate(history)
    holdout=uniform_cap(jax.random.key(seed+500),20000)
    candidates=[history[-1],history[-500:].mean(axis=0)]
    candidate_losses=[float(loss(jnp.asarray(t),holdout)) for t in candidates]
    chosen=int(np.argmin(candidate_losses));theta=candidates[chosen]
    mu,r,ho=unpack(jnp.asarray(theta))
    info={'fixed_latitude':LAT,'dimension':D,'df':2,'optimized_parameter_count':201,
          'mu':np.asarray(mu).tolist(),'R':float(r),'horizontal_observer':np.asarray(ho).tolist(),
          'mu_norm':float(jnp.linalg.norm(mu)),'horizontal_observer_norm':float(jnp.linalg.norm(ho)),
          'mu_along_skewness':float(jnp.dot(mu,core.AXIS)),
          'horizontal_observer_cosine_with_skewness':float(jnp.dot(ho,core.AXIS)/jnp.linalg.norm(ho)),
          'observer_distance_from_sphere_center':float(jnp.sqrt(jnp.sum(ho*ho)+B*B)),
          'seed':seed,'steps':steps,'batch_size':batch,'holdout_size':20000,'holdout_seed':seed+500,
          'initial_holdout_loss':float(loss(theta0,holdout)),'candidate_holdout_losses':candidate_losses,
          'chosen_candidate':['last','last_500_parameter_average'][chosen],
          'learning_rates':[.01,.002],'learning_rate_change_after_step':3000,
          'adam_beta1':.9,'adam_beta2':.999,'adam_epsilon':1e-8,
          'objective':'KL(pushforward uniform fixed bright cap || target), up to constants',
          'observer_parameterization':'sqrt(1-(latitude-1)^2) * tanh(||v||) * v/||v||',
          'elapsed_seconds':time.perf_counter()-begun}
    np.save(ROOT/'scs_theta.npy',theta)
    np.savez_compressed(ROOT/'scs_vi_history.npz',loss=losses,theta=history)
    (ROOT/'scs_parameters.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k not in ('mu','horizontal_observer')},indent=2),flush=True)

def tune():
    theta=jnp.asarray(np.load(ROOT/'scs_theta.npy'))
    rng=np.random.default_rng(26093002);x=core.exact_target(rng,40000)
    z=np.asarray(inverse(jnp.asarray(x),theta))
    target=jax.jit(lambda zz:logpi_sphere(zz,theta));lp=np.asarray(target(z))
    noise=rng.normal(size=z.shape);noise-=np.sum(noise*z,axis=1,keepdims=True)*z
    rows=[]
    def evaluate(h,record=True):
        p=z+h*noise;p/=np.linalg.norm(p,axis=1,keepdims=True)
        valid=p[:,-1]<B;safe=np.where(valid[:,None],p,z)
        a=valid*np.exp(np.minimum(0,np.asarray(target(safe))-lp))
        out={'step':float(h),'acceptance':float(a.mean()),'acceptance_se':float(a.std(ddof=1)/math.sqrt(len(a))),
             'boundary_rejection':float(1-valid.mean()),'native_esjd':float(np.mean(a*np.sum((p-z)**2,axis=1)))}
        if record:rows.append(out)
        return out
    grid=[.01,.02,.03,.05,.08,.10,.15,.2,.3,.5,.75,1.,1.5,2.,3.,5.,10.,14.,30.]
    for h in grid:evaluate(h)
    # Follow the previous grid-selection rule. Refine only when the grid misses
    # the 40% goal by more than one percentage point, using acceptance alone.
    selected=min(rows,key=lambda r:abs(r['acceptance']-.4))
    if abs(selected['acceptance']-.4)>.01:
        brackets=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if (a['acceptance']-.4)*(b['acceptance']-.4)<0]
        if brackets:
            a,b=brackets[0]
            for _ in range(12):
                middle=evaluate(math.sqrt(a['step']*b['step']))
                if (a['acceptance']-.4)*(middle['acceptance']-.4)<=0:b=middle
                else:a=middle
            selected=min(rows,key=lambda r:abs(r['acceptance']-.4))
    with (ROOT/'scs_stationary_step_tuning.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Independent post-selection acceptance estimate, not used in choosing h.
    rng2=np.random.default_rng(26093003);x2=core.exact_target(rng2,40000)
    z2=np.asarray(inverse(jnp.asarray(x2),theta));g=rng2.normal(size=z2.shape)
    g-=np.sum(g*z2,axis=1,keepdims=True)*z2
    pp=z2+selected['step']*g;pp/=np.linalg.norm(pp,axis=1,keepdims=True)
    valid=pp[:,-1]<B;aa=valid*np.exp(np.minimum(0,np.asarray(target(np.where(valid[:,None],pp,z2)))-np.asarray(target(z2))))
    result={'scs':selected,'tuning_seed':26093002,'stationary_draws':40000,'target_acceptance':.4,
            'independent_check_seed':26093003,'independent_check_acceptance':float(aa.mean()),
            'independent_check_se':float(aa.std(ddof=1)/math.sqrt(len(aa))),
            'independent_check_dark_side_rejection':float(1-valid.mean())}
    (ROOT/'scs_step_selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def make_runner(theta,h,max_steps=LIMIT):
    @jax.jit
    def run(key,z1,z2):
        initial=(jnp.int32(0),z1,z2,logpi_sphere(z1,theta),logpi_sphere(z2,theta),key,jnp.array(False),jnp.int32(0),jnp.int32(0),jnp.int32(0),jnp.int32(0))
        def cond(s):return (s[0]<max_steps)&(~s[6])
        def body(s):
            t,a,b,la,lb,key,met,ac1,ac2,br,pm=s
            key,kn,km,ku,kr=jax.random.split(key,5)
            g=jax.random.normal(kn,a.shape,dtype=jnp.float64)
            v=h*(g-a*jnp.dot(a,g));p=(a+v)/jnp.linalg.norm(a+v)
            ratio=core.spherical_logq(b,p,h)-core.spherical_logq(a,p,h)
            common=jnp.log(jax.random.uniform(km,dtype=jnp.float64))<=jnp.minimum(0.,ratio)
            e=(a-b)/jnp.maximum(jnp.linalg.norm(a-b),1e-150)
            reflected=p-2*e*jnp.dot(e,p);reflected/=jnp.linalg.norm(reflected)
            q=jnp.where(common,p,reflected)
            ok1=p[-1]<B;ok2=q[-1]<B
            lp=logpi_sphere(jnp.where(ok1,p,a),theta);lq=logpi_sphere(jnp.where(ok2,q,b),theta)
            u=jnp.log(jax.random.uniform(ku,dtype=jnp.float64))
            accept1=ok1&(u<=jnp.minimum(0.,lp-la));accept2=ok2&(u<=jnp.minimum(0.,lq-lb))
            a=jnp.where(accept1,p,a);b=jnp.where(accept2,q,b)
            return t+1,a,b,jnp.where(accept1,lp,la),jnp.where(accept2,lq,lb),key,jnp.all(a==b),ac1+accept1.astype(jnp.int32),ac2+accept2.astype(jnp.int32),br+(~ok1).astype(jnp.int32)+(~ok2).astype(jnp.int32),pm+common.astype(jnp.int32)
        s=jax.lax.while_loop(cond,body,initial)
        return s[0],s[6],s[7],s[8],s[9],s[10],s[1],s[2]
    return run

def validate():
    theta=jnp.asarray(np.load(ROOT/'scs_theta.npy'));rng=np.random.default_rng(26093004)
    x=np.concatenate([np.load(ROOT/'initial_states.npz')['initial_states'].reshape(-1,D),core.exact_target(rng,1000)])
    z=inverse(jnp.asarray(x),theta);back=np.asarray(map_logj(z,theta)[0])
    relative=float(np.max(np.linalg.norm(back-x,axis=1)/(1+np.linalg.norm(x,axis=1))))
    norm_error=float(np.max(abs(np.asarray(jnp.linalg.norm(z,axis=1))-1)))
    assert relative<1e-8 and norm_error<1e-10,(relative,norm_error)
    assert bool(jnp.all(z[:,-1]<B))
    base=uniform_cap(jax.random.key(26093005),512)
    axialmap=float(jnp.max(abs(map_logj(base,initial_theta())[0]-core.scs_map(base))))
    axiallog=float(jnp.max(abs(logpi_sphere(base,initial_theta())-core.scs_logpi(base))))
    assert axialmap<1e-9 and axiallog<1e-9
    jacerrors=[]
    for zz in np.asarray(base[:5]):
        basis=np.linalg.svd(zz[None,:],full_matrices=True)[2][1:].T
        numeric=np.linalg.slogdet(np.asarray(jax.jacfwd(lambda a:map_logj(a,theta)[0])(jnp.asarray(zz)))@basis)[1]
        jacerrors.append(abs(numeric-float(map_logj(zz,theta)[1])))
    assert max(jacerrors)<1e-8,jacerrors
    direction=rng.normal(size=201);direction/=np.linalg.norm(direction)
    gradient=jax.grad(loss)(theta,base);analytic=float(jnp.dot(gradient,direction))
    eps=1e-5;numeric=float((loss(theta+eps*direction,base)-loss(theta-eps*direction,base))/(2*eps))
    assert np.isfinite(gradient).all() and abs(analytic-numeric)<1e-4,(analytic,numeric)
    h=json.loads((ROOT/'scs_step_selection.json').read_text())['scs']['step'] if (ROOT/'scs_step_selection.json').exists() else .15
    run=make_runner(theta,h,1000);keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(26092913),i))(jnp.arange(3))
    bat=jax.jit(jax.vmap(run))(keys,z[:3],z[3:6]);scalar=[run(keys[i],z[i],z[i+3]) for i in range(3)]
    eq=all(np.array_equal(np.asarray(bat[k]),np.array([np.asarray(s[k]) for s in scalar])) for k in range(6))
    assert eq
    # At the old parameters, this general-observer coupling reduces to the old kernel.
    original=np.load(ROOT/'initial_states.npz')['sub_cauchy_sphere_states'][:5]
    old=core.make_runner('scs',.15,max_steps=1000);new=make_runner(initial_theta(),.15,1000)
    axialcounts=True
    for i,(a,b) in enumerate(original):
        key=jax.random.fold_in(jax.random.key(26092913),i)
        one=old(key,a,b);two=new(key,a,b)
        axialcounts &= all(np.array_equal(np.asarray(one[k]),np.asarray(two[k])) for k in range(6))
    assert axialcounts
    out={'passed':True,'roundtrip_max_relative_error':relative,'sphere_norm_max_error':norm_error,
         'axial_map_max_error':axialmap,'axial_logtarget_max_error':axiallog,'surface_jacobian_max_log_error':max(jacerrors),
         'directional_gradient_finite_difference_error':abs(analytic-numeric),'batch_scalar_counts_equal':eq,
         'axial_kernel_matches_original_counts':bool(axialcounts),'all_3000_inverse_states_in_bright_cap':True}
    (ROOT/'scs_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

def simulate():
    theta=jnp.asarray(np.load(ROOT/'scs_theta.npy'));h=json.loads((ROOT/'scs_step_selection.json').read_text())['scs']['step']
    x=np.load(ROOT/'initial_states.npz')['initial_states'];z=inverse(jnp.asarray(x),theta)
    np.savez_compressed(ROOT/'scs_fitted_initial_states.npz',sphere_states=np.asarray(z))
    runner=jax.jit(jax.vmap(make_runner(theta,h)));rows=[];begun=time.perf_counter()
    destination=ROOT/'scs_vi_meeting_times.csv'
    if destination.exists():rows=list(csv.DictReader(destination.open()))
    seen={int(r['replicate']) for r in rows}
    for start in range(0,1000,16):
        ids=list(range(start,min(start+16,1000)))
        if all(i in seen for i in ids):continue
        padded=ids+[ids[-1]]*(16-len(ids));idx=jnp.asarray(padded,dtype=jnp.int32)
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(26092913),i))(idx)
        tick=time.perf_counter();vals=[np.asarray(a) for a in runner(keys,z[idx,0],z[idx,1])[:6]]
        elapsed=time.perf_counter()-tick
        for j,i in enumerate(ids):
            if i in seen:continue
            it,met,a,b,br,pm=[v[j].item() for v in vals]
            assert it>0 and (met or it==LIMIT)
            rows.append({'method':'scs','replicate':i,'met':bool(met),'meeting_time':it if met else -1,'iterations':it,
                         'acceptance_rate_chain1':a/it,'acceptance_rate_chain2':b/it,'proposal_meetings':pm,'boundary_rejections':br,
                         'elapsed_seconds':elapsed/len(ids),'provenance':'new_scs_vi_1000','seed':26092913})
        rows.sort(key=lambda r:int(r['replicate']))
        with destination.with_suffix('.tmp').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
        destination.with_suffix('.tmp').replace(destination)
        print('SCS VI completed',len(rows),'of 1000; elapsed',time.perf_counter()-begun,flush=True)
    assert len(rows)==1000

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['fit','tune','validate','run','all'],default='all');args=parser.parse_args()
    if args.stage in ('fit','all'):fit()
    if args.stage in ('tune','all'):tune()
    if args.stage in ('validate','all'):validate()
    if args.stage in ('run','all'):simulate()
