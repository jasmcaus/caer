#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


import numpy as np
from numpy.core.multiarray import normalize_axis_index
from . import cndsupport
from . import cndi


__all__ = [
    'fourier_gaussian', 
    'fourier_uniform', 
    'fourier_ellipsoid',
    'fourier_shift'
]


def _get_output_fourier(output, inp):
    if output is None:
        if inp.dtype.type in [np.complex64, np.complex128,
                                np.float32]:
            output = np.zeros(inp.shape, dtype=inp.dtype)
        else:
            output = np.zeros(inp.shape, dtype=np.float64)

    elif type(output) is type:
        if output not in [np.complex64, np.complex128,
                          np.float32, np.float64]:
            raise RuntimeError("output type not supported")
        output = np.zeros(inp.shape, dtype=output)

    elif output.shape != inp.shape:
        raise RuntimeError("output shape not correct")

    return output


def _get_output_fourier_complex(output, inp):
    if output is None:
        if inp.dtype.type in [np.complex64, np.complex128]:
            output = np.zeros(inp.shape, dtype=inp.dtype)
        else:
            output = np.zeros(inp.shape, dtype=np.complex128)

    elif type(output) is type:
        if output not in [np.complex64, np.complex128]:
            raise RuntimeError("output type not supported")
        output = np.zeros(inp.shape, dtype=output)

    elif output.shape != inp.shape:
        raise RuntimeError("output shape not correct")
    return output


def fourier_gaussian(inp, sigma, n=-1, axis=-1, output=None):
    """
    Multidimensional Gaussian fourier filter.

    The array is multiplied with the fourier transform of a Gaussian
    kernel.

    Parameters
    ----------
    inp : array_like
        The inp array.
    sigma : float or sequence
        The sigma of the Gaussian kernel. If a float, `sigma` is the same for
        all axes. If a sequence, `sigma` has to contain one value for each
        axis.
    n : int, optional
        If `n` is negative (default), then the inp is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the inp is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : Tensor, optional
        If given, the result of filtering the inp is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_gaussian : Tensor
        The filtered inp.
    """
    inp = np.asarray(inp)
    output = _get_output_fourier(output, inp)
    axis = normalize_axis_index(axis, inp.ndim)
    sigmas = cndsupport._normalize_sequence(sigma, inp.ndim)
    sigmas = np.asarray(sigmas, dtype=np.float64)

    if not sigmas.flags.contiguous:
        sigmas = sigmas.copy()

    cndi.fourier_filter(inp, sigmas, n, axis, output, 0)
    return output


def fourier_uniform(inp, size, n=-1, axis=-1, output=None):
    """
    Multidimensional uniform fourier filter.

    The array is multiplied with the Fourier transform of a box of given
    size.

    Parameters
    ----------
    inp : array_like
        The inp array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the inp is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the inp is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : Tensor, optional
        If given, the result of filtering the inp is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_uniform : Tensor
        The filtered inp.
    """
    inp = np.asarray(inp)
    output = _get_output_fourier(output, inp)
    axis = normalize_axis_index(axis, inp.ndim)
    sizes = cndsupport._normalize_sequence(size, inp.ndim)
    sizes = np.asarray(sizes, dtype=np.float64)

    if not sizes.flags.contiguous:
        sizes = sizes.copy()

    cndi.fourier_filter(inp, sizes, n, axis, output, 1)
    return output


def fourier_ellipsoid(inp, size, n=-1, axis=-1, output=None):
    """
    Multidimensional ellipsoid Fourier filter.

    The array is multiplied with the fourier transform of a ellipsoid of
    given sizes.

    Parameters
    ----------
    inp : array_like
        The inp array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the inp is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the inp is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : Tensor, optional
        If given, the result of filtering the inp is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_ellipsoid : Tensor
        The filtered inp.

    Notes
    -----
    This function is implemented for arrays of rank 1, 2, or 3.
    """
    inp = np.asarray(inp)
    output = _get_output_fourier(output, inp)
    axis = normalize_axis_index(axis, inp.ndim)
    sizes = cndsupport._normalize_sequence(size, inp.ndim)
    sizes = np.asarray(sizes, dtype=np.float64)

    if not sizes.flags.contiguous:
        sizes = sizes.copy()
        
    cndi.fourier_filter(inp, sizes, n, axis, output, 2)
    return output


def fourier_shift(inp, shift, n=-1, axis=-1, output=None):
    """
    Multidimensional Fourier shift filter.

    The array is multiplied with the Fourier transform of a shift operation.

    Parameters
    ----------
    inp : array_like
        The inp array.
    shift : float or sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the inp is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the inp is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : Tensor, optional
        If given, the result of shifting the inp is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_shift : Tensor
        The shifted inp.
    """

    inp = np.asarray(inp)
    output = _get_output_fourier_complex(output, inp)
    axis = normalize_axis_index(axis, inp.ndim)
    shifts = cndsupport._normalize_sequence(shift, inp.ndim)
    shifts = np.asarray(shifts, dtype=np.float64)

    if not shifts.flags.contiguous:
        shifts = shifts.copy()

    cndi.fourier_shift(inp, shifts, n, axis, output)
    return output
