import ctypes as ct,time,sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LIB=ROOT/'native_build'/('stereographic.dylib' if sys.platform=='darwin' else 'stereographic.so')
LIB.parent.mkdir(exist_ok=True)
subprocess.run(['clang++' if sys.platform=='darwin' else 'g++','-O3','-std=c++17','-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(ROOT/'stereographic_native.cpp'),'-o',str(LIB)],check=True)
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import t as student_t
import coupling_experiment as c
import jax,jax.numpy as jnp
lib=ct.CDLL(str(LIB));ptr=ct.POINTER(ct.c_double);iptr=ct.POINTER(ct.c_int);i64=ct.POINTER(ct.c_int64)
lib.native_logtcdf.argtypes=[ct.c_double];lib.native_logtcdf.restype=ct.c_double
lib.native_logtarget.argtypes=[ptr];lib.native_logtarget.restype=ct.c_double
lib.native_step.argtypes=[ptr,ptr,ptr,ct.c_double,ct.c_double,ct.c_double,ptr,ptr,iptr]
lib.native_run.argtypes=[ptr,ptr,ct.c_uint64,ct.c_int,ct.c_double,i64]
def p(a):return a.ctypes.data_as(ptr)
rng=np.random.default_rng(26092915)
args=np.r_[-np.geomspace(1e-12,1500,1000),0,np.geomspace(1e-12,1500,1000)]
err=np.max(abs(np.array([lib.native_logtcdf(float(t)) for t in args])-student_t.logcdf(args,102)))
assert err<1e-9,err
x=np.load(ROOT/'initial_states.npz')['initial_states']
z=np.asarray(c.stereo_inverse(jnp.asarray(x)))
errtarget=max(abs(lib.native_logtarget(p(a))-float(c.stereo_logpi(a))) for a in z[:20].reshape(-1,101))
assert errtarget<1e-8,errtarget
@jax.jit
def reference(a,b,g,w,u):
 v=14*(g-a*jnp.dot(a,g));pp=(a+v)/jnp.linalg.norm(a+v)
 common=jnp.log(w)<=jnp.minimum(0.,c.spherical_logq(b,pp,14)-c.spherical_logq(a,pp,14))
 e=(a-b)/jnp.linalg.norm(a-b);q=pp-2*e*jnp.dot(e,pp);q=q/jnp.linalg.norm(q);q=jnp.where(common,pp,q)
 aa=jnp.log(u)<=jnp.minimum(0.,c.stereo_logpi(pp)-c.stereo_logpi(a));ab=jnp.log(u)<=jnp.minimum(0.,c.stereo_logpi(q)-c.stereo_logpi(b))
 a=jnp.where(aa,pp,a);b=jnp.where(ab,q,b)
 return a,b,jnp.array([jnp.all(a==b),aa,ab,common],dtype=jnp.int32)
maximum=0
for i in range(200):
 a,b=np.array(z[i]);g=rng.normal(size=101);w,u=rng.random(2);aout=np.empty(101);bout=np.empty(101);flags=np.empty(4,dtype=np.int32)
 lib.native_step(p(a),p(b),p(g),w,u,14,p(aout),p(bout),flags.ctypes.data_as(iptr))
 ar,br,fr=reference(a,b,g,w,u)
 assert np.array_equal(flags,np.asarray(fr)),(i,flags,fr)
 maximum=max(maximum,np.max(abs(aout-ar)),np.max(abs(bout-br)))
assert maximum<1e-12,maximum
output={'logcdf_max_error_vs_scipy':float(err),'target_max_error_vs_jax':float(errtarget),'matched_single_step_decisions':200,'max_single_step_state_error':float(maximum)}
for i in [0,100]:
 a,b=np.array(z[i]);out=np.empty(5,dtype=np.int64);start=time.perf_counter()
 lib.native_run(p(a),p(b),26092933+i,20000,14,out.ctypes.data_as(i64))
 print('benchmark',i,time.perf_counter()-start,out,flush=True)
json.dump(output,open(ROOT/'native_validation.json','w'),indent=2)
print(output,flush=True)
