"""
Yoshida
-------

Yoshida weights, coefficients and multistep sequence construnction (list of weighted seed steps)

"""
from typing import Any
from typing import Sequence
from typing import Callable
from typing import Optional

from jax import Array

from functools import reduce
from itertools import groupby

from sympint.util import first
from sympint.util import last


def weights(n:int) -> list[float]:
    """
    Generate Yoshida weights for a given Yoshida order

    Parameters
    ----------
    n: int, positive
        Yoshida order

    Returns
    -------
    list[float]

    Note
    ----
    The resulting integration step difference order is two times the Yoshida order
    Given a time-reversible integration step w(2n)(dt) with difference order 2n
    2(n + 1) order symmetric integration step w(2(n + 1))(dt) can be constructed using Yoshida weights for (n + 1)
    x1, x2, x1 = weights(n + 1)
    w(2(n + 1))(dt) = w(2n)(x1 dt) o w(2n)(x2 dt) o w(2n)(x1 dt)

    """
    return [
        +1/(2 - 2**(1/(1 + 2*n))),
        -2**(1/(1 + 2*n))/(2 - 2**(1/(1 + 2*n))),
        +1/(2 - 2**(1/(1 + 2*n)))
    ]


def coefficients(ni:int,  nf:int) -> list[float]:
    """
    Generate Yoshida coefficients for given Yoshida orders (ni < nf)

    Parameters
    ----------
    ni: int, non-negative
        initial Yoshida order
    nf: int, non-negative
        final Yoshida order

    Returns
    -------
    list[float]

    Note
    ----
    Given a time-reversible integration step w(2(ni - 1))(dt)
    Construct coefficients x1, x2, x3, ..., x3, x2, x1, so that
    w(2 nf)(dt) = w(2(ni - 1))(x1 dt) o w(2(ni - 1))(x2 dt) o ... o w(2(ni - 1))(x2 dt) o w(2(ni - 1))(x1 dt)

    """
    ws = map(weights, range(ni if ni != 0 else 1, nf + 1))
    return reduce(lambda xs, x: [xi*xsi for xi in x for xsi in xs], ws, [1.0])


def table(k:int, ni:int, nf:int, merge:bool=False) -> tuple[list[int], list[float]]:
    """
    Generate Yoshida multistep table (sequence of coefficients for multistep integrator)

    Parameters
    ----------
    k: int, positive
        number of mappings
    ni: int, non-negative
        initial Yoshida order
    nf: int, non-negative
        final Yoshida order
    merge: bool, default=False
        flag to merge edge mappings (assume commuting)

    Returns
    -------
    tuple[list[int], list[float]]

    Note
    ----
    Given a set of symplectic mappings indexed as 0, 1, ..., (l - 1) and Yoshida ni <= nf orders
    Construct Yoshida coefficients (i1, x1), (i2, x2), ..., (i2, x2), (i1, x1)
    w(2m)(dt) = w(i1)(x1 dx) o w(i2)(x2 dx) o ... o w(i2)(x2 dx) o w(i1)(x1 dx)

    """
    ps = [[i, 0.5] for i in range(k - 1)]
    ps = ps + [[k - 1, 1.0]] + [*reversed(ps)]
    ns, vs = map(list, zip(*ps))
    cs = coefficients(ni, nf)
    ps = sum(([[n, v*c] for (n, v) in zip(ns, vs)] for c in cs), start = [])
    if merge:
        gs = groupby(ps, key=lambda x: first(x))
        ps = [reduce(lambda x, y: [first(x), last(x) + last(y)], g) for _, g in gs]
    return tuple(*map(list, zip(*ps)))

def sequence(ni:int,
             nf:int,
             mappings:Sequence[Callable[..., Array]],
             merge:bool=False,
             parameters:Optional[list[list[Any]]]=None) ->  Sequence[Callable[..., Array]]:
    """
    Construct Yoshida integrator multistep sequence (ordered sequence of weighted mappings)

    Parameters
    ----------
    ni: int, non-negative
        initial Yoshida order
    nf: int, non-negative
        final Yoshida order
    mappings: Sequence[Callable[..., Array]]
        list of (time-reversible) mappings
    merge: bool, default=False
        flag to merge edge mappings (assume commuting)
    parameters: Optional[list[list[Any]]], default=None
        list of optional fixed parameters for each mapping

    Returns
    -------
    Sequence[Callable[..., Array]]

    Note
    ----
    Each input mapping is assumed to have (x, dt, *args, *pars) signature
    Output sequence mappings have (x, dt, *args) singnatures

    """
    indices, weights = table(len(mappings), ni, nf, merge)
    parameters = [[] for _ in range(len(mappings))] if parameters is None else parameters
    parameters = [parameters[i] for i in indices]
    def wrapper(mapping, weight, parameter):
        def function(x, dt, *args):
            return mapping(x, weight*dt, *args, *parameter)
        return function
    return [wrapper(mappings[index], weight, parameter) for (index, weight, parameter) in zip(indices, weights, parameters)]
