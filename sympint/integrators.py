"""
Integrators
-----------

Collection of symplectic (JAX composable) integrators

"""
from typing import Callable

import jax
from jax import Array

from sympint.functional import nest
from sympint.functional import fold

from sympint.yoshida import sequence

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


def tao(H:Callable[..., Array], binding:float=0.0) -> Callable[..., Array]:
    """
    Generate Tao integrator

    Parameters
    ----------
    H: Callable[[Array, *Any], Array]
        Hamiltonian function H(q, p, dt, *args)
    binding: float, default=0.0
        binding factor

    Returns
    -------
    Callable[[Array, *Any], Array]
        integrator(qp, dt, *args)

    """
    dHdq = jax.grad(H, argnums=0)
    dHdp = jax.grad(H, argnums=1)
    def fa(state:Array, dt:Array, *args:Array) -> Array:
        q, p, Q, P = state.reshape(4, -1)
        return jax.numpy.concatenate([q, p - dt*dHdq(q, P, *args), Q + dt*dHdp(q, P, *args), P])
    def fb(state:Array, dt:Array, *args:Array) -> Array:
        q, p, Q, P = state.reshape(4, -1)
        return jax.numpy.concatenate([q + dt*dHdp(Q, p, *args), p, Q, P - dt*dHdq(Q, p, *args)])
    def fc(state:Array, dt:Array, *args:Array) -> Array:
        q, p, Q, P = state.reshape(4, -1)
        omega = 2*binding*dt
        cos = jax.numpy.cos(omega)
        sin = jax.numpy.sin(omega)
        dq = q - Q
        dp = p - P
        return jax.numpy.concatenate([
            0.5*(q + Q + cos*dq + sin*dp),
            0.5*(p + P - sin*dq + cos*dp),
            0.5*(q + q - cos*dq - sin*dp),
            0.5*(p + P + sin*dq - cos*dp)
        ])
    step = fold(sequence(0, 0, [fa, fb, fc] if binding != 0.0 else [fa, fb], merge=True))
    def integrator(state:Array, dt:Array, *args:Array) -> Array:
        local = step(jax.numpy.concatenate([state, state]), dt, *args)
        q, p, *_ = local.reshape(4, -1)
        return jax.numpy.concatenate([q, p])
    return integrator
