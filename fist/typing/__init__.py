"""
============================
Typing (:mod:`numpy.typing`)
============================

.. warning::

  Some of the types in this module rely on features only present in
  the standard library in Python 3.8 and greater. If you want to use
  these types in earlier versions of Python, you should install the
  typing-extensions_ package.

Large parts of the NumPy API have PEP-484-style type annotations. In
addition, the following type aliases are available for users.

- ``typing.ArrayLike``: objects that can be converted to arrays
- ``typing.DtypeLike``: objects that can be converted to dtypes

Roughly speaking, ``typing.ArrayLike`` is "objects that can be used as
inputs to ``np.array``" and ``typing.DtypeLike`` is "objects that can
be used as inputs to ``np.dtype``".

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The ``ArrayLike`` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at 0x10c004cd0>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as ``Any``:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at 0x1192741d0>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool_

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should insted use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

dtype
~~~~~

The ``DTypeLike`` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid Numpy code, the type checker will complain about it,
since its usage is discouraged.
Please see : https://numpy.org/devdocs/reference/arrays.dtypes.html

NBitBase
~~~~~~~~

.. autoclass:: numpy.typing.NBitBase

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 8):
        from typing import final
    else:
        from typing_extensions import final
else:
    def final(f): return f


@final  # Dissallow the creation of arbitrary `NBitBase` subclasses
class NBitBase:
    """
    An object representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierachieral set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating a
    function that takes a float and integer of arbitrary precision as arguments
    and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from typing import TypeVar, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> T = TypeVar("T", bound=npt.NBitBase)

        >>> def add(a: "np.floating[T]", b: "np.integer[T]") -> "np.floating[T]":
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
            "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()


# Silence errors about subclassing a `@final`-decorated class
class _256Bit(NBitBase): ...  # type: ignore[misc]
class _128Bit(_256Bit): ...  # type: ignore[misc]
class _96Bit(_128Bit): ...  # type: ignore[misc]
class _80Bit(_96Bit): ...  # type: ignore[misc]
class _64Bit(_80Bit): ...  # type: ignore[misc]
class _32Bit(_64Bit): ...  # type: ignore[misc]
class _16Bit(_32Bit): ...  # type: ignore[misc]
class _8Bit(_16Bit): ...  # type: ignore[misc]

# Clean up the namespace
del TYPE_CHECKING, final

from ._scalars import (
    _CharLike,
    _BoolLike,
    _IntLike,
    _FloatLike,
    _ComplexLike,
    _NumberLike,
    _ScalarLike,
    _VoidLike,
)
from ._array_like import _SupportsArray, ArrayLike
from ._shape import _Shape, _ShapeLike
from ._dtype_like import _SupportsDtype, _VoidDtypeLike, DtypeLike

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
