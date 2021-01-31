#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import numpy as np 

def _check_target_size(size):
    """
    Common check to enforce type and sanity check on size tuples
    :param size: Should be a tuple of size 2 (width, height)
    :returns: True, or raises a ValueError
    """

    if not isinstance(size, (list, tuple)):
        raise ValueError("`target_size` must be a tuple of length 2 `(width,height)`")
    if len(size) != 2:
        raise ValueError("`target_size` must be a tuple of length 2 `(width,height)`")
    if size[0] < 0 or size[1] < 0:
        raise ValueError("Width and height must be >= 0")

    return True


def _check_mean_sub_values(value, channels):
    """
        Checks if mean subtraction values are valid based on the number of channels
        'value' must be a tuple of dimensions = number of channels
    Returns boolean:
        True -> Expression is valid
        False -> Expression is invalid
    """
    if value is None:
        raise ValueError('Value(s) specified is of NoneType()')
    
    if isinstance(value, tuple):
        # If not a tuple, we convert it to one
        try:
            value = tuple(value)
        except TypeError:
            value = tuple([value])
    
    if channels not in [1,3]:
        raise ValueError('Number of channels must be either 1 (Grayscale) or 3 (RGB/BGR)')

    if len(value) not in [1,3]:
        raise ValueError('Tuple length must be either 1 (subtraction over the entire image) or 3 (per channel subtraction)', value)
    
    if len(value) == channels:
        return True 

    else:
        raise ValueError(f'Expected a tuple of dimension {channels}', value) 


def _get_output(array, out, fname, dtype=None, output=None):
    '''
    output = _get_output(array, out, fname, dtype=None, output=None)
    Implements the caer output convention:
        (1) if `out` is None, return np.empty(array.shape, array.dtype)
        (2) else verify that output is of right size, shape, and contiguous
    Parameters
    ----------
    array : Tensor
    out : Tensor or None
    fname : str
        Function name. Used in error messages
    Returns
    -------
    output : Tensor
    '''
    detail = '.\nWhen an output argument is used, the checking is very strict as this is a performance feature.'
    if dtype is None:
        dtype = array.dtype
    if output is not None: # pragma: no cover
        import warnings
        warnings.warn('Using deprecated `output` argument in function `%s`. Please use `out` in the future. It has exactly the same meaning and it matches what numpy uses.' % fname, DeprecationWarning)
        if out is not None:
            warnings.warn('Using both `out` and `output` in function `%s`.\ncaer is going to ignore the `output` argument and use the `out` version exclusively.' % fname)
        else:
            out = output

    if out is None:
        return np.empty(array.shape, dtype)

    if out.dtype != dtype:
        raise ValueError(
            'caer.%s: `out` has wrong type (out.dtype is %s; expected %s)%s' %
                (fname, out.dtype, dtype, detail))

    if out.shape != array.shape:
        raise ValueError('caer.%s: `out` has wrong shape (got %s, while expecting %s)%s' % (fname, out.shape, array.shape, detail))

    if not out.flags.contiguous:
        raise ValueError('caer.%s: `out` is not c-array%s' % (fname,detail))

    return out

def _get_axis(array, axis, fname):
    '''
    axis = _get_axis(array, axis, fname)
    Checks that ``axis`` is a valid axis of ``array`` and normalises it.
    Parameters
    ----------
    array : Tensor
    axis : int
    fname : str
        Function name. Used in error messages
    Returns
    -------
    axis : int
        The positive index of the axis to use
    '''
    if axis < 0:
        axis += len(array.shape)

    if not (0 <= axis < len(array.shape)):
        raise ValueError('caer.%s: `axis` is out of bounds (maximum was %s, got %s)' % (fname, array.ndim, axis))

    return axis


def _normalize_sequence(array, value, fname):
    '''
    values = _normalize_sequence(array, value, fname)
    If `value` is a sequence, checks that it has an element for each dimension
    of `array`. Otherwise, returns a sequence that repeats `value` once for
    each dimension of array.
    Parameters
    ----------
    array : Tensor
    value : sequence or scalar
    fname : str
        Function name. Used in error messages
    Returns
    -------
    values : sequence
    '''
    try:
        value = list(value)
    except TypeError:
        return [value for s in array.shape]
    if len(value) != array.ndim:
        raise ValueError('caer.%s: argument is sequence, but has wrong size (%s for an array of %s dimensions)' % (fname, len(value), array.ndim))
    return value


def _verify_is_floatingpoint_type(A, function_name):
    '''
    _verify_is_integer_type(array, "function")
    Checks that ``A`` is a floating-point array. If it is not, it raises
    ``TypeError``.
    Parameters
    ----------
    A : Tensor
    function_name : str
        Used for error messages
    '''
    if not np.issubdtype(A.dtype, np.floating):
        raise TypeError('caer.{}: This function only accepts floating-point types (passed array of type {})'.format(function_name, A.dtype))


def _verify_is_integer_type(A, function_name):
    '''
    _verify_is_integer_type(array, "function")
    Checks that ``A`` is an integer array. If it is not, it raises
    ``TypeError``.
    Parameters
    ----------
    A : Tensor
    function_name : str
        Used for error messages
    '''
    k = A.dtype.kind
    if k not in "iub": # integer, unsigned integer, boolean
        raise TypeError('caer.%s: This function only accepts integer types (passed array of type %s)' % (function_name, A.dtype))


def _verify_is_nonnegative_integer_type(A, function_name):
    '''
    _verify_is_nonnegative_integer_type(array, "function")
    Checks that ``A`` is an unsigned integer array. If it is not, it raises
    ``TypeError``.
    Parameters
    ----------
    A : Tensor
    function_name : str
        Used for error messages
    '''
    _verify_is_integer_type(A, function_name)
    if A.dtype.kind == 'i' and not np.all(A >= 0):
        raise ValueError('caer.{0}: This function only accepts positive integer types (passed array of type {1})'.format(function_name, A.dtype))


def _make_binary(array):
    '''
    bin = _make_binary(array)
    Returns (possibly a copy) of array as a boolean array
    '''
    array = np.asanyarray(array)
    if array.dtype != bool:
        return (array != 0)
    return array


def _as_floating_point_array(array):
    '''
    array = _as_floating_point_array(array)
    Returns (possibly a copy) of array as a floating-point array
    '''
    array = np.asanyarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        return array.astype(np.double)
    return array


def _check_3(arr, funcname):
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError('caer.%s: this function expects an array of shape (h, w, 3), received an array of shape %s.' % (funcname, arr.shape))


def _check_2(arr, funcname):
    if arr.ndim != 2:
        raise ValueError('caer.%s: this function can only handle 2D arrays (passed array with shape %s).' % (funcname, arr.shape))