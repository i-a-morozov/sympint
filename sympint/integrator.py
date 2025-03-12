"""
Integrator
----------

Collection of symplectic (JAX composable) integrators

"""
from typing import Callable

import jax
from jax import Array

from sympint.functional import nest

def midpoint(H:Callable[..., Array], ns:int=1) -> Callable[..., Array]:
    """
    Generate implicit midpoint integrator

    Parameters
    ----------
    H: Callable[[Array, *Any], Array]
        Hamiltonian function H(q, p, dt, t, *args)
    ns: int, default=1
        number of Newton iteration steps

    Returns
    -------
    Callable[[Array, *Any], Array]
        integrator(qp, dt, t, *args)

    """
    dHdq = jax.grad(H, argnums=0)
    dHdp = jax.grad(H, argnums=1)
    def integrator(state: Array, dt: Array, t: Array, *args: Array) -> Array:
        q, p = jax.numpy.reshape(state, (2, -1))
        t_m = t + 0.5*dt
        def residual(state: Array) -> tuple[Array, Array]:
            Q, P = jax.numpy.reshape(state, (2, -1))
            q_m = 0.5*(q + Q)
            p_m = 0.5*(p + P)
            dq = Q - q - dt*dHdp(q_m, p_m, t_m, *args)
            dp = P - p + dt*dHdq(q_m, p_m, t_m, *args)
            state = jax.numpy.concatenate([dq, dp])
            return state, state
        def newton(state: Array) -> Array:
            jacobian, error = jax.jacrev(residual, has_aux=True)(state)
            delta, *_ = jax.numpy.linalg.lstsq(jacobian, -error)
            return state + delta
        return nest(ns, newton)(state)
    return integrator