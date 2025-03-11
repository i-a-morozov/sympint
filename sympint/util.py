
"""
Util
----

Utils module

"""
from typing import Any
from typing import Iterable

def first(xs:Iterable[Any]) -> Any:
    """
    Return first element

    Parameters
    ----------
    xs : Iterable[Any]
        xs

    Returns
    -------
    Any

    Examples
    --------
    >>> first([1, 2, 3, 4])
    1

    """
    x, *_ = xs
    return x


def last(xs:Iterable[Any]) -> Any:
    """
    Return last element

    Parameters
    ----------
    xs : Iterable[Any]
        xs

    Returns
    -------
    Any

    Examples
    --------
    >>> last([1, 2, 3, 4])
    4

    """
    *_, x = xs
    return x