#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


from collections.abc import Iterable
import warnings
import numpy as np
from numpy.core.multiarray import normalize_axis_index

from . import cndsupport
from . import cndi


__all__ = [
    'correlate1d', 
    'convolve1d', 
    'gaussian_filter1d', 
    'gaussian_filter',     
    'prewitt',
    'sobel', 
    'generic_laplace', 
    'laplace',         
    'gaussian_laplace', 
    'generic_gradient_magnitude',   
    'gaussian_gradient_magnitude', 
    'correlate', 
    'convolve', 
    'median_filter',   
    'generic_filter1d', 
    'generic_filter'
]


def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)


def _complex_via_real_components(func, inp, weights, output, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_inp = inp.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'

    if complex_inp and complex_weights:
        # real component of the output
        func(inp.real, weights.real, output=output.real, **kwargs)
        output.real -= func(inp.imag, weights.imag, output=None, **kwargs)
        # imaginary component of the output
        func(inp.real, weights.imag, output=output.imag, **kwargs)
        output.imag += func(inp.imag, weights.real, output=None, **kwargs)

    elif complex_inp:
        func(inp.real, weights, output=output.real, **kwargs)
        func(inp.imag, weights, output=output.imag, **kwargs)

    else:
        func(inp, weights.real, output=output.real, **kwargs)
        func(inp, weights.imag, output=output.imag, **kwargs)

    return output


def correlate1d(inp, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(inp)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    inp = np.asarray(inp)
    weights = np.asarray(weights)

    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'

    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, cval=cval, origin=origin)
        output = cndsupport._get_output(output, input, complex_output=True)

        return _complex_via_real_components(correlate1d, input, weights,
                                            output, **kwargs)

    output = cndsupport._get_output(output, inp)
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')

    if not weights.flags.contiguous:
        weights = weights.copy()

    axis = normalize_axis_index(axis, inp.ndim)

    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')

    mode = cndsupport._extend_mode_to_code(mode)
    cndi.correlate1d(inp, weights, axis, output, mode, cval,
                          origin)
    return output



def convolve1d(inp, weights, axis=-1, output=None, mode="reflect",
               cval=0.0, origin=0):
    """Calculate a 1-D convolution along the given axis.

    The lines of the array along the given axis are convolved with the
    given weights.

    Parameters
    ----------
    %(inp)s
    weights : Tensor
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s

    Returns
    -------
    convolve1d : Tensor
        Convolved array with same shape as inp
    """
    weights = weights[::-1]
    origin = -origin

    if not len(weights) & 1:
        origin -= 1

    weights = np.asarray(weights)

    if weights.dtype.kind == 'c':
        # pre-conjugate here to counteract the conjugation in correlate1d
        weights = weights.conj()

    return correlate1d(inp, weights, axis, output, mode, cval, origin)



def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')

    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x

    else:
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P

        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        
        return q * phi_x



def gaussian_filter1d(inp, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(inp)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter1d : Tensor
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

    return correlate1d(inp, weights, axis, output, mode, cval, 0)



def gaussian_filter(inp, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.

    Parameters
    ----------
    %(inp)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter : Tensor
        Returned array of same shape as `inp`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.
    """
    inp = np.asarray(inp)
    output = cndsupport._get_output(output, inp)
    orders = cndsupport._normalize_sequence(order, inp.ndim)
    sigmas = cndsupport._normalize_sequence(sigma, inp.ndim)
    modes = cndsupport._normalize_sequence(mode, inp.ndim)
    axes = list(range(inp.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]

    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(inp, sigma, axis, order, output,
                              mode, cval, truncate)
            inp = output

    else:
        output[...] = inp[...]

    return output



def prewitt(inp, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Prewitt filter.

    Parameters
    ----------
    %(inp)s
    %(axis)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    """
    inp = np.asarray(inp)
    axis = normalize_axis_index(axis, inp.ndim)
    output = cndsupport._get_output(output, inp)
    modes = cndsupport._normalize_sequence(mode, inp.ndim)
    correlate1d(inp, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(inp.ndim) if ii != axis]

    for ii in axes:
        correlate1d(output, [1, 1, 1], ii, output, modes[ii], cval, 0,)

    return output



def sobel(inp, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Sobel filter.

    Parameters
    ----------
    %(inp)s
    %(axis)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    """
    inp = np.asarray(inp)
    axis = normalize_axis_index(axis, inp.ndim)
    output = cndsupport._get_output(output, inp)
    modes = cndsupport._normalize_sequence(mode, inp.ndim)
    correlate1d(inp, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(inp.ndim) if ii != axis]

    for ii in axes:
        correlate1d(output, [1, 2, 1], ii, output, modes[ii], cval, 0)

    return output



def generic_laplace(inp, derivative2, output=None, mode="reflect",
                    cval=0.0,
                    extra_arguments=(),
                    extra_keywords=None):
    """
    N-D Laplace filter using a provided second derivative function.

    Parameters
    ----------
    %(inp)s
    derivative2 : callable
        Callable with the following signature::

            derivative2(inp, axis, output, mode, cval,
                        *extra_arguments, **extra_keywords)

        See `extra_arguments`, `extra_keywords` below.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
    """
    if extra_keywords is None:
        extra_keywords = {}

    inp = np.asarray(inp)
    output = cndsupport._get_output(output, inp)
    axes = list(range(inp.ndim))

    if len(axes) > 0:
        modes = cndsupport._normalize_sequence(mode, len(axes))
        derivative2(inp, axes[0], output, modes[0], cval,
                    *extra_arguments, **extra_keywords)

        for ii in range(1, len(axes)):
            tmp = derivative2(inp, axes[ii], output.dtype, modes[ii], cval,
                              *extra_arguments, **extra_keywords)
            output += tmp

    else:
        output[...] = inp[...]

    return output



def laplace(inp, output=None, mode="reflect", cval=0.0):
    """N-D Laplace filter based on approximate second derivatives.

    Parameters
    ----------
    %(inp)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    """
    def derivative2(inp, axis, output, mode, cval):
        return correlate1d(inp, [1, -2, 1], axis, output, mode, cval, 0)

    return generic_laplace(inp, derivative2, output, mode, cval)



def gaussian_laplace(inp, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs):
    """Multidimensional Laplace filter using Gaussian second derivatives.

    Parameters
    ----------
    %(inp)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    Extra keyword arguments will be passed to gaussian_filter().
    """
    inp = np.asarray(inp)

    def derivative2(inp, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * inp.ndim
        order[axis] = 2
        return gaussian_filter(inp, sigma, order, output, mode, cval,
                               **kwargs)

    return generic_laplace(inp, derivative2, output, mode, cval,
                           extra_arguments=(sigma,),
                           extra_keywords=kwargs)



def generic_gradient_magnitude(inp, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
    """Gradient magnitude using a provided gradient function.

    Parameters
    ----------
    %(inp)s
    derivative : callable
        Callable with the following signature::

            derivative(inp, axis, output, mode, cval,
                       *extra_arguments, **extra_keywords)

        See `extra_arguments`, `extra_keywords` below.
        `derivative` can assume that `inp` and `output` are Tensors.
        Note that the output from `derivative` is modified inplace;
        be careful to copy important inps before returning them.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
    """
    if extra_keywords is None:
        extra_keywords = {}

    inp = np.asarray(inp)
    output = cndsupport._get_output(output, inp)
    axes = list(range(inp.ndim))

    if len(axes) > 0:
        modes = cndsupport._normalize_sequence(mode, len(axes))
        derivative(inp, axes[0], output, modes[0], cval,
                   *extra_arguments, **extra_keywords)
        np.multiply(output, output, output)

        for ii in range(1, len(axes)):
            tmp = derivative(inp, axes[ii], output.dtype, modes[ii], cval,
                             *extra_arguments, **extra_keywords)
            np.multiply(tmp, tmp, tmp)
            output += tmp
        # This allows the sqrt to work with a different default casting
        np.sqrt(output, output, casting='unsafe')

    else:
        output[...] = inp[...]

    return output



def gaussian_gradient_magnitude(inp, sigma, output=None,
                                mode="reflect", cval=0.0, **kwargs):
    """Multidimensional gradient magnitude using Gaussian derivatives.

    Parameters
    ----------
    %(inp)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    Extra keyword arguments will be passed to gaussian_filter().

    Returns
    -------
    gaussian_gradient_magnitude : Tensor
        Filtered array. Has the same shape as `inp`.
    """
    inp = np.asarray(inp)

    def derivative(inp, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * inp.ndim
        order[axis] = 1
        return gaussian_filter(inp, sigma, order, output, mode,
                               cval, **kwargs)

    return generic_gradient_magnitude(inp, derivative, output, mode,
                                      cval, extra_arguments=(sigma,),
                                      extra_keywords=kwargs)


def _correlate_or_convolve(inp, weights, output, mode, cval, origin,
                           convolution):
    inp = np.asarray(inp)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'

    if complex_input or complex_weights:
        if complex_weights and not convolution:
            # As for numpy.correlate, conjugate weights rather than input.
            weights = weights.conj()
        kwargs = dict(
            mode=mode, cval=cval, origin=origin, convolution=convolution
        )
        output = cndsupport._get_output(output, input, complex_output=True)

        return _complex_via_real_components(_correlate_or_convolve, input,
                                            weights, output, **kwargs)

    origins = cndsupport._normalize_sequence(origin, inp.ndim)
    weights = np.asarray(weights, dtype=np.float64)
    wshape = [ii for ii in weights.shape if ii > 0]

    if len(wshape) != inp.ndim:
        raise RuntimeError('filter weights array has incorrect shape.')

    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origins)):
            origins[ii] = -origins[ii]
            if not weights.shape[ii] & 1:
                origins[ii] -= 1

    for origin, lenw in zip(origins, wshape):
        if _invalid_origin(origin, lenw):
            raise ValueError('Invalid origin; origin must satisfy '
                             '-(weights.shape[k] // 2) <= origin[k] <= '
                             '(weights.shape[k]-1) // 2')

    if not weights.flags.contiguous:
        weights = weights.copy()

    output = cndsupport._get_output(output, inp)
    temp_needed = np.may_share_memory(inp, output)

    if temp_needed:
        # inp and output arrays cannot share memory
        temp = output
        output = cndsupport._get_output(output.dtype, inp)

    if not isinstance(mode, str) and isinstance(mode, Iterable):
        raise RuntimeError("A sequence of modes is not supported")

    mode = cndsupport._extend_mode_to_code(mode)
    cndi.correlate(inp, weights, output, mode, cval, origins)

    if temp_needed:
        temp[...] = output
        output = temp

    return output



def correlate(inp, weights, output=None, mode='reflect', cval=0.0,
              origin=0):
    """
    Multidimensional correlation.

    The array is correlated with the given kernel.

    Parameters
    ----------
    %(inp)s
    weights : Tensor
        array of weights, same number of dimensions as inp
    %(output)s
    %(mode)s
    %(cval)s
    %(origin_multiple)s

    Returns
    -------
    result : Tensor
        The result of correlation of `inp` with `weights`.
    """
    return _correlate_or_convolve(inp, weights, output, mode, cval,
                                  origin, False)



def convolve(inp, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    """
    Multidimensional convolution.

    The array is convolved with the given kernel.

    Parameters
    ----------
    %(inp)s
    weights : array_like
        Array of weights, same number of dimensions as inp
    %(output)s
    %(mode)s
    cval : scalar, optional
        Value to fill past edges of inp if `mode` is 'constant'. Default
        is 0.0
    %(origin_multiple)s

    Returns
    -------
    result : Tensor
        The result of convolution of `inp` with `weights`.

    See Also
    --------
    correlate : Correlate an image with a kernel.

    Notes
    -----
    Each value in result is :math:`C_i = \\sum_j{I_{i+k-j} W_j}`, where
    W is the `weights` kernel,
    j is the N-D spatial index over :math:`W`,
    I is the `inp` and k is the coordinate of the center of
    W, specified by `origin` in the inp parameters.
    """
    return _correlate_or_convolve(inp, weights, output, mode, cval,
                                  origin, True)


def median_filter(inp, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
    """
    Calculate a multidimensional median filter.

    Parameters
    ----------
    %(inp)s
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin_multiple)s

    Returns
    -------
    median_filter : Tensor
        Filtered array. Has the same shape as `inp`.
    """
    return _rank_filter(inp, 0, size, footprint, output, mode, cval,
                        origin, 'median')


def minimum_filter1d(inp, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D minimum filter along the given axis.
    The lines of the array along the given axis are filtered with a
    minimum filter of given size.
    Parameters
    ----------
    %(inp)s
    size : int
        length along which to calculate 1D minimum
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s
    Notes
    -----
    This function implements the MINLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `inp` length, regardless of filter size.
    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html
    """
    inp = np.asarray(inp)

    if np.iscomplexobj(inp):
        raise TypeError('Complex type not supported')

    axis = normalize_axis_index(axis, inp.ndim)

    if size < 1:
        raise RuntimeError('incorrect filter size')

    output = cndsupport._get_output(output, inp)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')

    mode = cndsupport._extend_mode_to_code(mode)
    cndi.min_or_max_filter1d(inp, size, axis, output, mode, cval,
                                  origin, 1)
    return output


def maximum_filter1d(inp, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D maximum filter along the given axis.
    The lines of the array along the given axis are filtered with a
    maximum filter of given size.
    Parameters
    ----------
    %(inp)s
    size : int
        Length along which to calculate the 1-D maximum.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s
    Returns
    -------
    maximum1d : Tensor, None
        Maximum-filtered array with same shape as inp.
        None if `output` is not None
    Notes
    -----
    This function implements the MAXLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `inp` length, regardless of filter size.
    """
    inp = np.asarray(inp)

    if np.iscomplexobj(inp):
        raise TypeError('Complex type not supported')

    axis = normalize_axis_index(axis, inp.ndim)

    if size < 1:
        raise RuntimeError('incorrect filter size')

    output = cndsupport._get_output(output, inp)

    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')

    mode = cndsupport._extend_mode_to_code(mode)
    cndi.min_or_max_filter1d(inp, size, axis, output, mode, cval,
                                  origin, 0)
    return output


def _min_or_max_filter(inp, size, footprint, structure, output, mode,
                       cval, origin, minimum):
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)

    if structure is None:
        if footprint is None:
            if size is None:
                raise RuntimeError("no footprint provided")
            separable = True

        else:
            footprint = np.asarray(footprint, dtype=bool)
            if not footprint.any():
                raise ValueError("All-zero footprint is not supported.")
            if footprint.all():
                size = footprint.shape
                footprint = None
                separable = True
            else:
                separable = False

    else:
        structure = np.asarray(structure, dtype=np.float64)
        separable = False
        if footprint is None:
            footprint = np.ones(structure.shape, bool)
        else:
            footprint = np.asarray(footprint, dtype=bool)

    inp = np.asarray(inp)

    if np.iscomplexobj(inp):
        raise TypeError('Complex type not supported')

    output = cndsupport._get_output(output, inp)
    temp_needed = np.may_share_memory(inp, output)

    if temp_needed:
        # inp and output arrays cannot share memory
        temp = output
        output = cndsupport._get_output(output.dtype, inp)
    origins = cndsupport._normalize_sequence(origin, inp.ndim)

    if separable:
        sizes = cndsupport._normalize_sequence(size, inp.ndim)
        modes = cndsupport._normalize_sequence(mode, inp.ndim)
        axes = list(range(inp.ndim))
        axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
                for ii in range(len(axes)) if sizes[ii] > 1]

        if minimum:
            filter_ = minimum_filter1d
        else:
            filter_ = maximum_filter1d

        if len(axes) > 0:
            for axis, size, origin, mode in axes:
                filter_(inp, int(size), axis, output, mode, cval, origin)
                inp = output
        else:
            output[...] = inp[...]

    else:
        fshape = [ii for ii in footprint.shape if ii > 0]

        if len(fshape) != inp.ndim:
            raise RuntimeError('footprint array has incorrect shape.')

        for origin, lenf in zip(origins, fshape):
            if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
                raise ValueError('invalid origin')

        if not footprint.flags.contiguous:
            footprint = footprint.copy()

        if structure is not None:
            if len(structure.shape) != inp.ndim:
                raise RuntimeError('structure array has incorrect shape')

            if not structure.flags.contiguous:
                structure = structure.copy()

        if not isinstance(mode, str) and isinstance(mode, Iterable):
            raise RuntimeError(
                "A sequence of modes is not supported for non-separable "
                "footprints")

        mode = cndsupport._extend_mode_to_code(mode)
        cndi.min_or_max_filter(inp, footprint, structure, output, mode, cval, origins, minimum)

    if temp_needed:
        temp[...] = output
        output = temp

    return output



def minimum_filter(inp, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    """Calculate a multidimensional minimum filter.
    Parameters
    ----------
    %(inp)s
    %(size_foot)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(origin_multiple)s
    Returns
    -------
    minimum_filter : Tensor
        Filtered array. Has the same shape as `inp`.
    """
    return _min_or_max_filter(inp, size, footprint, None, output, mode, cval, origin, 1)



def maximum_filter(inp, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    """Calculate a multidimensional maximum filter.
    Parameters
    ----------
    %(inp)s
    %(size_foot)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(origin_multiple)s
    Returns
    -------
    maximum_filter : Tensor
        Filtered array. Has the same shape as `inp`.
    """
    return _min_or_max_filter(inp, size, footprint, None, output, mode, cval, origin, 0)


def _rank_filter(inp, rank, size=None, footprint=None, output=None,
                 mode="reflect", cval=0.0, origin=0, operation='rank'):
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)

    inp = np.asarray(inp)

    if np.iscomplexobj(inp):
        raise TypeError('Complex type not supported')

    origins = cndsupport._normalize_sequence(origin, inp.ndim)

    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")

        sizes = cndsupport._normalize_sequence(size, inp.ndim)
        footprint = np.ones(sizes, dtype=bool)

    else:
        footprint = np.asarray(footprint, dtype=bool)

    fshape = [ii for ii in footprint.shape if ii > 0]

    if len(fshape) != inp.ndim:
        raise RuntimeError('filter footprint array has incorrect shape.')

    for origin, lenf in zip(origins, fshape):
        if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
            raise ValueError('invalid origin')

    if not footprint.flags.contiguous:
        footprint = footprint.copy()

    filter_size = np.where(footprint, 1, 0).sum()

    if operation == 'median':
        rank = filter_size // 2

    elif operation == 'percentile':
        percentile = rank
        if percentile < 0.0:
            percentile += 100.0

        if percentile < 0 or percentile > 100:
            raise RuntimeError('invalid percentile')

        if percentile == 100.0:
            rank = filter_size - 1
        else:
            rank = int(float(filter_size) * percentile / 100.0)

    if rank < 0:
        rank += filter_size

    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')

    if rank == 0:
        return minimum_filter(inp, None, footprint, output, mode, cval, origins)

    elif rank == filter_size - 1:
        return maximum_filter(inp, None, footprint, output, mode, cval, origins)

    else:
        output = cndsupport._get_output(output, inp)
        temp_needed = np.may_share_memory(inp, output)

        if temp_needed:
            # inp and output arrays cannot share memory
            temp = output
            output = cndsupport._get_output(output.dtype, inp)

        if not isinstance(mode, str) and isinstance(mode, Iterable):
            raise RuntimeError( "A sequence of modes is not supported by non-separable rank filters")

        mode = cndsupport._extend_mode_to_code(mode)
        cndi.rank_filter(inp, rank, footprint, output, mode, cval,
                              origins)

        if temp_needed:
            temp[...] = output
            output = temp

        return output

def generic_filter1d(inp, function, filter_size, axis=-1,
                     output=None, mode="reflect", cval=0.0, origin=0,
                     extra_arguments=(), extra_keywords=None):
    """Calculate a 1-D filter along the given axis.

    `generic_filter1d` iterates over the lines of the array, calling the
    given function at each line. The arguments of the line are the
    inp line, and the output line. The inp and output lines are 1-D
    double arrays. The inp line is extended appropriately according
    to the filter size and origin. The output line must be modified
    in-place with the result.

    Parameters
    ----------
    %(inp)s
    function : {callable, scipy.LowLevelCallable}
        Function to apply along given axis.
    filter_size : scalar
        Length of the filter.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    %(extra_arguments)s
    %(extra_keywords)s
    """
    if extra_keywords is None:
        extra_keywords = {}

    inp = np.asarray(inp)

    if np.iscomplexobj(inp):
        raise TypeError('Complex type not supported')

    output = cndsupport._get_output(output, inp)

    if filter_size < 1:
        raise RuntimeError('invalid filter size')

    axis = normalize_axis_index(axis, inp.ndim)

    if (filter_size // 2 + origin < 0) or (filter_size // 2 + origin >=
                                           filter_size):
        raise ValueError('invalid origin')

    mode = cndsupport._extend_mode_to_code(mode)
    cndi.generic_filter1d(inp, function, filter_size, axis, output,
                               mode, cval, origin, extra_arguments,
                               extra_keywords)
    return output



def generic_filter(inp, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0,
                   extra_arguments=(), extra_keywords=None):
    """Calculate a multidimensional filter using the given function.

    At each element the provided function is called. The inp values
    within the filter footprint at that element are passed to the function
    as a 1-D array of double values.

    Parameters
    ----------
    %(inp)s
    function : {callable, scipy.LowLevelCallable}
        Function to apply at each element.
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin_multiple)s
    %(extra_arguments)s
    %(extra_keywords)s
    """
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)

    if extra_keywords is None:
        extra_keywords = {}

    inp = np.asarray(inp)

    if np.iscomplexobj(inp):
        raise TypeError('Complex type not supported')

    origins = cndsupport._normalize_sequence(origin, inp.ndim)

    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = cndsupport._normalize_sequence(size, inp.ndim)
        footprint = np.ones(sizes, dtype=bool)

    else:
        footprint = np.asarray(footprint, dtype=bool)

    fshape = [ii for ii in footprint.shape if ii > 0]

    if len(fshape) != inp.ndim:
        raise RuntimeError('filter footprint array has incorrect shape.')

    for origin, lenf in zip(origins, fshape):
        if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
            raise ValueError('invalid origin')

    if not footprint.flags.contiguous:
        footprint = footprint.copy()

    output = cndsupport._get_output(output, inp)
    mode = cndsupport._extend_mode_to_code(mode)
    cndi.generic_filter(inp, function, footprint, output, mode,
                             cval, origins, extra_arguments, extra_keywords)
                             
    return output
