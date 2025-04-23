# sympint, 2025

JAX composable symplectic integrators

[![DOI](https://zenodo.org/badge/946107787.svg)](https://doi.org/10.5281/zenodo.15265929)

# Install

```
$ pip install git+https://github.com/i-a-morozov/sympint.git@main
```

# Documentation

[https://i-a-morozov.github.io/sympint/](https://i-a-morozov.github.io/sympint/)

# Demo


```python
# In this demo construction of symplectic integrators is illustated for basic accelerator elements
```


```python
# Import

import torch
import jax

# Exact solutions

from model.library.transformations import drift
from model.library.transformations import quadrupole
from model.library.transformations import bend

# Function iterations

from sympint import nest
from sympint import fold

# Integrators and composer

from sympint import sequence
from sympint import midpoint
from sympint import tao
```


```python
# Set data type

jax.config.update("jax_enable_x64", True)
```


```python
# Set device

device, *_ = jax.devices('cpu')
jax.config.update('jax_default_device', device)
```


```python
# Define Hamiltonial functions for accelerator elements

def h_drif(qs, ps, t, dp, *args):
    qx, qy = qs
    px, py = ps
    return 1/2*(px**2 + py**2)/(1 + dp)

def h_quad(qs, ps, t, kn, ks, dp, *args):
    qx, qy = qs
    px, py = ps
    return 1/2*(px**2 + py**2)/(1 + dp) + 1/2*kn*(qx**2 - qy**2) - ks*qx*qy

def h_bend(qs, ps, t, rd, kn, ks, dp, *args):
    qx, qy = qs
    px, py = ps
    return 1/2*(px**2 + py**2)/(1 + dp) - qx*dp/rd + qx**2/(2*rd**2) + 1/2*kn*(qx**2 - qy**2) - ks*qx*qy
```


```python
# Set parameters

ti = torch.tensor(0.0, dtype=torch.float64)
dt = torch.tensor(0.1, dtype=torch.float64)
rd = torch.tensor(25.0, dtype=torch.float64)
kn = torch.tensor(2.0, dtype=torch.float64)
ks = torch.tensor(0.1, dtype=torch.float64)
dp = torch.tensor(0.001, dtype=torch.float64)
```


```python
# Hamiltonian conservation (drif)

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
hi = h_drif(qs, ps, ti, dp)

(qx, px, qy, py) = drift(x, dp, dt)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
hf = h_drif(qs, ps, ti, dp)

print(torch.allclose(hi, hf, rtol=1.0E-16, atol=1.0E-16))
```

    True



```python
# Hamiltonian conservation (quad)

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
hi = h_quad(qs, ps, ti, kn, ks, dp)

(qx, px, qy, py) = quadrupole(x, kn, ks, dp, dt)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
hf = h_quad(qs, ps, ti, kn, ks, dp)

print(torch.allclose(hi, hf, rtol=1.0E-16, atol=1.0E-16))
```

    True



```python
# Hamiltonian conservation (bend)

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
hi = h_bend(qs, ps, ti, rd, kn, ks, dp)

(qx, px, qy, py) = bend(x, rd, kn, ks, dp, dt)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
hf = h_bend(qs, ps, ti, rd, kn, ks, dp)

print(torch.allclose(hi, hf, rtol=1.0E-16, atol=1.0E-16))
```

    True



```python
# To illustrate (multi-map) split and (Yoshida) composition explicit symplectic integrator consider the following split
# h = h1 + h2 = 1/2*(px**2 + py**2)/(1 + dp) - qx*dp/rd + qx**2/(2*rd**2) + 1/2*kn*(qx**2 - qy**2) - ks*qx*qy
# h1 = 1/2*(px**2 + py**2)/(1 + dp)
# qx = qx + dt*px/(1 + dp)
# px = px
# qy = qy + dt*py/(1 + dp)
# py = py
# h2 = - qx*dp/rd + qx**2/(2*rd**2) + 1/2*kn*(qx**2 - qy**2) - ks*qx*qy
# qx = qx
# px = px + dt*(dp/rd - qx/rd**2 - kn*qx + ks*qy)
# qy = qy
# py = py + dt*(kn*qy + ks*qx)

def fa(x, dt, rd, kn, ks, dp):
    qx, qy, px, py = x
    return jax.numpy.stack([qx + dt*px/(1 + dp), qy + dt*py/(1 + dp), px, py])

def fb(x, dt, rd, kn, ks, dp):
    qx, qy, px, py = x
    return jax.numpy.stack([qx, qy, px + dt*(dp/rd - qx/rd**2 - kn*qx + ks*qy), py + dt*(kn*qy + ks*qx)])
```


```python
# Yoshida (bend)

# Generate integration step

step = fold(sequence(0, 1, [fa, fb], merge=True))

# Evaluate integration step

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
qsps = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = qsps.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))

qsps = step(qsps, dt.item(), rd.item(), kn.item(), ks.item(), dp.item())
qs, ps = qsps.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))
print()

# Evaluate exact solution

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
QsPs = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = QsPs.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))

(qx, px, qy, py) = bend(x, rd, kn, ks, dp, dt)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
QsPs = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = QsPs.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))
print()

# Compare

print(qsps)
print(QsPs)
print(jax.numpy.linalg.norm(qsps - QsPs))
```

    8.030437562437563e-05
    8.030437539389926e-05
    
    8.030437562437563e-05
    8.03043756243756e-05
    
    [ 0.00999747 -0.0049949  -0.00105058 -0.0003978 ]
    [ 0.00999746 -0.00499491 -0.00105066 -0.00039784]
    9.384719084898185e-08



```python
# Midpoint (bend)

# Generate integration step

step = fold(sequence(0, 1, [midpoint(h_bend, ns=1)], merge=False))

# Evaluate integration step

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
qsps = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = qsps.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))

qsps = step(qsps, dt.item(), ti.item(), rd.item(), kn.item(), ks.item(), dp.item())
qs, ps = qsps.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))
print()

# Evaluate exact solution

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
QsPs = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = QsPs.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))

(qx, px, qy, py) = bend(x, rd, kn, ks, dp, dt)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
QsPs = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = QsPs.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))
print()

# Compare

print(qsps)
print(QsPs)
print(jax.numpy.linalg.norm(qsps - QsPs))
```

    8.030437562437563e-05
    8.030437562437561e-05
    
    8.030437562437563e-05
    8.03043756243756e-05
    
    [ 0.00999747 -0.0049949  -0.00105061 -0.00039781]
    [ 0.00999746 -0.00499491 -0.00105066 -0.00039784]
    5.8710763609389174e-08



```python
# Tao (bend)

# Generate integration step

step = fold(sequence(0, 1, [tao(h_bend, binding=0.0)], merge=False))

# Evaluate integration step

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
qsps = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = qsps.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))

qsps = step(qsps, dt.item(), ti.item(), rd.item(), kn.item(), ks.item(), dp.item())
qs, ps = qsps.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))
print()

# Evaluate exact solution

(qx, px, qy, py) = x = torch.tensor([0.01, 0.001, -0.005, 0.0005], dtype=torch.float64)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
QsPs = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = QsPs.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))

(qx, px, qy, py) = bend(x, rd, kn, ks, dp, dt)
qs = torch.stack([qx, qy])
ps = torch.stack([px, py])
QsPs = jax.numpy.array(torch.hstack([qs, ps]).tolist())
qs, ps = QsPs.reshape(2, -1)
print(h_bend(qs, ps, ti.item(), rd.item(), kn.item(), ks.item(), dp.item()))
print()

# Compare

print(qsps)
print(QsPs)
print(jax.numpy.linalg.norm(qsps - QsPs))
```

    8.030437562437563e-05
    8.030437585721544e-05
    
    8.030437562437563e-05
    8.03043756243756e-05
    
    [ 0.00999747 -0.0049949  -0.00105064 -0.00039783]
    [ 0.00999746 -0.00499491 -0.00105066 -0.00039784]
    2.50766882131807e-08



```python

```


