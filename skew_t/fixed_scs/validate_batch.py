import sys,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import coupling_experiment as c
import numpy as np,jax,jax.numpy as jnp
z=np.load(ROOT/'initial_states.npz');theta=jnp.asarray(np.load(ROOT/'dcs_theta.npy'))
checks={}
for method,h,seed in [('scs',.15,26092913),('dcs_bw',.14,26092923),('euclidean',.26,26092943),('stereographic',14.,26092933)]:
 x=jnp.asarray(z['initial_states'][:3])
 native={'scs':lambda:jnp.asarray(z['sub_cauchy_sphere_states'][:3]),'dcs_bw':lambda:c.inverse_ball(x,theta),'euclidean':lambda:x,'stereographic':lambda:c.stereo_inverse(x)}[method]()
 run=c.make_runner(method,h,theta,max_steps=1000)
 keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(seed),i))(jnp.arange(3))
 bat=jax.jit(jax.vmap(run))(keys,native[:,0],native[:,1])
 single=[run(keys[i],native[i,0],native[i,1]) for i in range(3)]
 equal=all(np.array_equal(np.asarray(bat[k]),np.array([np.asarray(s[k]) for s in single])) for k in range(6))
 err=max(float(np.max(np.abs(np.asarray(bat[k])-np.array([np.asarray(s[k]) for s in single])))) for k in [6,7])
 assert equal,(method,'batch count mismatch')
 assert err<1e-8,(method,err)
 checks[method]={'first_six_outputs_identical':equal,'maximum_final_state_difference':err}
json.dump(checks,open(ROOT/'batch_validation.json','w'),indent=2)
print(checks)
