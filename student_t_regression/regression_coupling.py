"""Readable NumPy reference kernels and portable native backend for Section 5.2."""
from pathlib import Path
import csv
import ctypes as ct
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
D, NU, R = 14, 4., np.sqrt(14.)
STEPS = {'stereographic': .007, 'euclidean': .03}

def load_data(path=ROOT / 'BostonHousing.csv'):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        features = [k for k in reader.fieldnames if k != 'medv']
        rows = list(reader)
    raw_x = np.array([[float(r[k]) for k in features] for r in rows])
    raw_y = np.array([float(r['medv']) for r in rows])
    xmean, xscale = raw_x.mean(0), raw_x.std(0, ddof=0)
    ymean, yscale = raw_y.mean(), raw_y.std(ddof=0)
    X = np.ascontiguousarray((raw_x-xmean)/xscale)
    y = np.ascontiguousarray((raw_y-ymean)/yscale)
    if X.shape != (506, 13):
        raise ValueError('This experiment requires 506 rows and 13 predictors.')
    return X, y, dict(features=features, xmean=xmean.tolist(), xscale=xscale.tolist(),
                     ymean=float(ymean), yscale=float(yscale))

def log_density(theta, X, y, nu=NU):
    """Flat beta prior, p(sigma^2) proportional to 1/sigma^2; u=log(sigma)."""
    r = y-X @ theta[:-1]
    with np.errstate(divide='ignore'):
        v = 2*np.log(np.abs(r))-np.log(nu)-2*theta[-1]
    return float(-.5*(nu+1)*np.logaddexp(0., v).sum()-len(y)*theta[-1])

def to_sphere(x, R=R):
    norm2 = np.sum(x*x, axis=-1, keepdims=True)
    return np.concatenate((2*R*x/(norm2+R*R), (norm2-R*R)/(norm2+R*R)), axis=-1)

def from_sphere(z, R=R):
    return R*z[..., :-1]/(1-z[..., -1:])

def sphere_log_density(z, X, y):
    return log_density(from_sphere(z), X, y)-D*np.log1p(-z[-1])

def proposal_log_density(a, p, h):
    c = min(float(a @ p), 1.)
    if c <= 0:
        return -np.inf
    return -(D+1)*np.log(c)-(1-c)*(1+c)/(2*h*h*c*c)

def coupled_proposals(a, b, g, w, method, h):
    """Maximal overlap test, then reflection; identical inputs stay identical."""
    if method == 'stereographic':
        p = a+h*(g-a*(a @ g)/(a @ a))
        p /= np.linalg.norm(p)
        common = np.array_equal(a,b) or np.log(w) <= min(0., proposal_log_density(b,p,h)-proposal_log_density(a,p,h))
        if common:
            return p,p.copy(),True
        e = a-b
        q = p-2*e*(e @ p)/(e @ e)
        return p,q/np.linalg.norm(q),False
    delta = (a-b)/h
    p = a+h*g
    common = np.array_equal(a,b) or np.log(w) <= min(0., -g @ delta-.5*(delta @ delta))
    if common:
        return p,p.copy(),True
    q = b+h*(g-2*delta*(delta @ g)/(delta @ delta))
    return p,q,False

def reference_step(a,b,g,w,u,method,h,X,y):
    p,q,common = coupled_proposals(a,b,g,w,method,h)
    target = sphere_log_density if method == 'stereographic' else log_density
    aa = np.log(u) <= min(0., target(p,X,y)-target(a,X,y))
    ab = np.log(u) <= min(0., target(q,X,y)-target(b,X,y))
    return (p if aa else a).copy(), (q if ab else b).copy(), np.array([aa,ab,common],int)

def build_library():
    build = ROOT/'native_build'
    build.mkdir(exist_ok=True)
    source = ROOT/'regression_native.cpp'
    mac = sys.platform == 'darwin'
    library = build/('regression.dylib' if mac else 'regression.so')
    if not library.exists() or source.stat().st_mtime > library.stat().st_mtime:
        subprocess.run(['clang++' if mac else 'g++','-O3','-std=c++17',
                        '-dynamiclib' if mac else '-shared','-fPIC',str(source),'-o',str(library)],check=True)
    lib = ct.CDLL(str(library))
    dp = ct.POINTER(ct.c_double)
    prefix = [dp,dp,ct.c_int,ct.c_double,ct.c_double,ct.c_int]
    lib.regression_logtarget.argtypes = prefix+[dp]
    lib.regression_logtarget.restype = ct.c_double
    lib.regression_step.argtypes = prefix+[dp,dp,dp,ct.c_double,ct.c_double,ct.c_double,dp,dp,ct.POINTER(ct.c_int)]
    lib.regression_step.restype = None
    lib.regression_run.argtypes = prefix+[dp,dp,ct.c_uint64,ct.c_int,ct.c_double,ct.c_double,ct.POINTER(ct.c_int64)]
    lib.regression_run.restype = None
    return lib

def ptr(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_double))

def splitmix64(x):
    mask = (1 << 64)-1
    x = (x+0x9E3779B97F4A7C15)&mask
    x = ((x^(x>>30))*0xBF58476D1CE4E5B9)&mask
    x = ((x^(x>>27))*0x94D049BB133111EB)&mask
    return x^(x>>31)
