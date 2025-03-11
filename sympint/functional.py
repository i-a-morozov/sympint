"""
Functional
----------

Collection of factories for functional iteration
Output of nest and fold function factories can be composed with regular JAX functions

"""
from typing import Any
from typing import Callable
from typing import Sequence

import jax
from jax import Array


def nest(length: int, mapping: Callable[..., Array]) -> Callable[..., Array]:
    """
    Create a function that iteratively applies a state transformation mapping

    Parameters
    ----------
    length : int, positive
        number of iterations to perform
    mapping : Callable[[Array, *Any], Array]
        state transformation mapping
        R^n x ... -> R^n

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    Nest is equivalent to the following Python loop:

    for _ in range(n):
        x = f(x, *args)

    """
    def closure(x: Array, *args: Any) -> Array:
        def scan_body(carry: tuple[Array, tuple], _: Any) -> tuple[tuple[Array, tuple], None]:
            x, args = carry
            x = mapping(x, *args)
            return (x, args), None
        (x, *_), _ = jax.lax.scan(scan_body, (x, args), None, length=length)
        return x
    return closure


def nest_list(length: int, mapping: Callable[..., Array]) -> Callable[..., Array]:
    """
    Create a function that iteratively applies a state transformation mapping
    And accumulates intermediate results

    Parameters
    ----------
    length : int, positive
        number of iterations to perform
    mapping : Callable[[Array, *Any], Array]
        state transformation mapping
        R^n x ... -> R^n

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    Initial value is not included in the output, output length is equal to the number of iterations
    Accumulate is equivalent to the following Python loop:

    xs = []
    for _ in range(n):
        x = f(x, *args)
        xs.append(x)

    """
    def closure(x: Array, *args: Any) -> Array:
        def scan_body(carry: tuple[Array, tuple], _: Any) -> tuple[tuple[Array, tuple], Array]:
            x, args = carry
            x = mapping(x, *args)
            return (x, args), x
        _, xs = jax.lax.scan(scan_body, (x, args), None, length=length)
        return xs
    return closure


def fold(mappings: Sequence[Callable[..., Array]]) -> Callable[..., Array]:
    """
    Create a function that sequentially applies mappings from a given list

    Parameters
    ----------
    mappings : Sequence[Callable[[Array, *Any], Array]]
        ordered sequence of transformation (identical signature) functions
        [R^n x ... -> R^n]

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    All mappings are assumed to have identical signature  f(x, *args) and R^n x ... -> R^n
    Fold over a list of mappings is equivalent to the following Python loop:

    for f in fs:
        x = f(x, *args)

    """
    idxs: Array = jax.numpy.arange(len(mappings))
    def closure(x: Array, *args: Any) -> Array:
        def scan_body(carry: tuple[Array, tuple], idx: Array) -> tuple[tuple[Array, tuple], None]:
            x, args = carry
            x = jax.lax.switch(idx, mappings, x, *args)
            return (x, args), None
        (x, *_), _ = jax.lax.scan(scan_body, (x, args), idxs)
        return x
    return closure


def fold_list(mappings: Sequence[Callable[..., Array]]) -> Callable[..., Array]:
    """
    Create a function that sequentially applies mappings from a given list
    And accumulates intermediate results

    Parameters
    ----------
    mappings : Sequence[Callable[[Array, *Any], Array]]
        ordered sequence of transformation (identical signature) functions
        [R^n x ... -> R^n]

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    All mappings are assumed to have identical signature  f(x, *args) and R^n x ... -> R^n
    Fold over a list of mappings is equivalent to the following Python loop:

    xs = []
    for f in fs:
        x = f(x, *args)
        xs.append(x)

    """
    idxs: Array = jax.numpy.arange(len(mappings))
    def closure(x: Array, *args: Any) -> Array:
        def scan_body(carry: tuple[Array, tuple], idx: Array) -> tuple[tuple[Array, tuple], Array]:
            x, args = carry
            x = jax.lax.switch(idx, mappings, x, *args)
            return (x, args), x
        _, xs = jax.lax.scan(scan_body, (x, args), idxs)
        return xs
    return closure
