# #    _____           ______  _____ 
# #  / ____/    /\    |  ____ |  __ \
# # | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# # | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# # | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
# #  \_____\/_/    \_ \______ |_|  \_\

# # Licensed under the MIT License <http://opensource.org/licenses/MIT>
# # SPDX-License-Identifier: MIT
# # Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


# import numpy as np
# from . import cconvolve
# from ..morph import cmorph 

# from .._internal import _get_output, _normalize_sequence, _verify_is_floatingpoint_type, _as_floating_point_array
# from .filter import mode2int, _check_mode


# __all__ = [
#     'daubechies',
#     'haar',
#     'median_filter',
#     'convolve',
#     'convolve1d',
#     'gaussian_filter',
#     'gaussian_filter1d',
#     'laplacian_2D'
# ]


# mode2int = {
#     'nearest' : 0,
#     'wrap' : 1,
#     'reflect' : 2,
#     'mirror' : 3,
#     'constant' : 4,
#     'ignore' : 5,
# }

# modes = frozenset(mode2int.keys())

# def _checked_mode2int(mode, cval, fname):
#     if mode not in modes:
#         raise ValueError(f'caer.{fname}: `mode` not in {modes}')

#     if mode == 'constant' and cval != 0.:
#         raise NotImplementedError('Please email caer developers to get this implemented.')
    
#     return mode2int[mode]

# _check_mode = _checked_mode2int


# def convolve(f, weights, mode='reflect', cval=0.0, out=None, output=None):
#     """
#     Convolution of `f` and `weights`

#     Convolution is performed in `doubles` to avoid over/underflow, but the
#     result is then cast to `f.dtype`. **This conversion may result in
#     over/underflow when using small integer types or unsigned types (if the
#     output is negative).** Converting to a floating point representation avoids
#     this issue::
#         c = convolve(f.astype(float), kernel)

#     Args:
#     f (Tensor): input. Any dimension is supported
#     weights (Tensor): weight filter. If not of the same dtype as `f`, it is cast
#     mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
#     cval (double, optional): If `mode` is constant, which constant to use (default: 0.0)
#     out (Tensor, optional): Output array. Must have same shape and dtype as `f` as well as be
#         C-contiguous.

#     Returns:
#     convolved : Tensor of same dtype as `f`
#     """
#     weights = weights.astype(f.dtype, copy=False)
#     if f.ndim != weights.ndim:
#         raise ValueError('caer.convolve: `f` and `weights` must have the same dimensions')
#     output = _get_output(f, out, 'convolve', output=output)
#     _check_mode(mode, cval, 'convolve')
#     return cconvolve.convolve(f, weights, output, mode2int[mode])


# def convolve1d(f, weights, axis, mode='reflect', cval=0., out=None):
#     """
#     Convolution of `f` and `weights` along axis `axis`.
#     Convolution is performed in `doubles` to avoid over/underflow, but the
#     result is then cast to `f.dtype`.

#     Args:
#         f (Tensor) : 
#             input. Any dimension is supported
#         weights (1-D Tensor) : 
#             weight filter. If not of the same dtype as `f`, it is cast
#         axis (int) : 
#             Axis along which to convolve
#         mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
#             How to handle borders
#         cval (double, optional) : 
#             If `mode` is constant, which constant to use (default: 0.0)
#         out (Tensor, optional) : 
#             Output array. Must have same shape and dtype as `f` as well as be
#             C-contiguous.

#     Returns:
#         convolved : Tensor of same dtype as `f`
#     """
#     weights = np.asanyarray(weights)
#     weights = weights.squeeze()
#     if weights.ndim != 1:
#         raise ValueError('caer.convolve1d: only 1-D sequences allowed')
#     _check_mode(mode, cval, 'convolve1d')
#     if f.flags.contiguous and len(weights) < f.shape[axis]:
#         weights = weights.astype(np.double, copy=False)
#         indices = [a for a in range(f.ndim) if a != axis] + [axis]
#         rindices = [indices.index(a) for a in range(f.ndim)]
#         oshape = f.shape
#         f = f.transpose(indices)
#         tshape = f.shape
#         f = f.reshape((-1, f.shape[-1]))

#         out = _get_output(f, out, 'convolve1d')
#         cconvolve.convolve1d(f, weights, out, mode2int[mode])
#         out = out.reshape(tshape)
#         out = out.transpose(rindices)
#         out = out.reshape(oshape)
#         return out
#     else:
#         index = [None] * f.ndim
#         index[axis] = slice(0, None)
#         weights = weights[tuple(index)]
#         return convolve(f, weights, mode=mode, cval=cval, out=out)


# def median_filter(f, Bc=None, mode='reflect', cval=0.0, out=None, output=None):
#     """
#     Median filter

#     Args:
#         f (Tensor) : 
#             input. Any dimension is supported
#         Bc (Tensor or int, optional) : 
#             Defines the neighbourhood, default is a square of side 3.
#         mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
#             How to handle borders
#         cval (double, optional) : 
#             If `mode` is constant, which constant to use (default: 0.0)
#         out (Tensor, optional) : 
#             Output array. Must have same shape and dtype as `f` as well as be
#             C-contiguous.

#     Returns:
#     median : Tensor of same type and shape as ``f``
#              median[i,j] is the median value of the points in f close to (i,j)
#     """
#     if Bc is None:
#         Bc = np.ones((3,) * len(f.shape), f.dtype)
#     elif f.dtype != Bc.dtype:
#         Bc = Bc.astype(f.dtype)
#     if f.ndim != Bc.ndim:
#         raise ValueError('caer.median_filter: `f` and `Bc` must have the same number of dimensions')
#     rank = Bc.sum()//2
#     output = _get_output(f, out, 'median_filter', output=output)
#     _check_mode(mode, cval, 'median_filter')
#     return cconvolve.rank_filter(f, Bc, output, int(rank), mode2int[mode])


# def mean_filter(f, Bc, mode='ignore', cval=0.0, out=None):
#     """
#     Mean filter. The value at ``mean[i,j]`` will be the mean of the values in
#     the neighbourhood defined by ``Bc``.

#     Args:
#         f (Tensor) : 
#             input. Any dimension is supported
#         Bc (Tensor) : 
#             Defines the neighbourhood. Must be explicitly passed, no default.
#         mode : {'reflect', 'nearest', 'wrap', 'mirror', 'constant', 'ignore' [ignore]}
#             How to handle borders. The default is to ignore points beyond the
#             border, so that the means computed near the border include fewer elements.
#         cval (double, optional) : 
#             If `mode` is constant, which constant to use (default: 0.0)
#         out (Tensor, optional) : 
#             Output array. Must be a double array with the same shape as `f` as well
#             as be C-contiguous.
#     Returns:
#         mean : Tensor of type double and same shape as ``f``
#     """
#     Bc = cmorph.get_structuring_elem(f, Bc)
#     out = _get_output(f, out, 'mean_filter', dtype=np.float64)
#     _check_mode(mode, cval, 'mean_filter')
#     return cconvolve.mean_filter(f, Bc, out, mode2int[mode], cval)


# def gaussian_filter1d(array, sigma, axis=-1, order=0, mode='reflect', cval=0., output=None):
#     """
#     One-dimensional Gaussian filter.

#     Args:
#         array (Tensor) : 
#             input array of a floating-point type
#         sigma (float) : 
#             standard deviation for Gaussian kernel (in pixel units)
#         axis (int, optional) : 
#             axis to operate on
#         order : {0, 1, 2, 3}, optional
#             An order of 0 corresponds to convolution with a Gaussian
#             kernel. An order of 1, 2, or 3 corresponds to convolution with
#             the first, second or third derivatives of a Gaussian. Higher
#             order derivatives are not implemented
#         mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
#             How to handle borders
#         cval (double, optional) : 
#             If `mode` is constant, which constant to use (default: 0.0)

#     Returns:
#         filtered (Tensor) : Filtered version of `array`
#     """
#     _verify_is_floatingpoint_type(array, 'gaussian_filter1d')
#     sigma = float(sigma)
#     s2 = sigma*sigma
#     # make the length of the filter equal to 4 times the standard
#     # deviations:
#     lw = int(4.0 * sigma + 0.5)

#     if lw <= 0:
#         raise ValueError('caer.gaussian_filter1d: sigma must be greater or equal to 0.125 [1/8]')

#     x = np.arange(2*lw+1, dtype=float)
#     x -= lw
#     weights = np.exp(x*x/(-2.*s2))
#     weights /= np.sum(weights)

#     # implement first, second and third order derivatives:
#     if order == 0:
#         pass
#     elif order == 1 : # first derivative
#         weights *= -x/s2
#     elif order == 2: # second derivative
#         weights *= (x*x/s2-1.)/s2
#     elif order == 3: # third derivative
#         weights *= (3.0 - x*x/s2)*x/(s2*s2)
#     else:
#         raise ValueError('caer.convolve.gaussian_filter1d: Order outside 0..3 not implemented')

#     return convolve1d(array, weights, axis, mode, cval, out=output)


# def gaussian_filter(array, sigma, order=0, mode='reflect', cval=0., out=None, output=None):
#     """
#     Multi-dimensional Gaussian filter.

#     Args:
#         array (Tensor) : 
#             input array, any dimension is supported. If the array is an integer
#             array, it will be converted to a double array.
#         sigma (scalar or sequence of scalars) : 
#             standard deviation for Gaussian kernel. The standard
#             deviations of the Gaussian filter are given for each axis as a
#             sequence, or as a single number, in which case it is equal for
#             all axes.
#         order : {0, 1, 2, 3} or sequence from same set, optional 
#             The order of the filter along each axis is given as a sequence
#             of integers, or as a single number.  An order of 0 corresponds
#             to convolution with a Gaussian kernel. An order of 1, 2, or 3
#             corresponds to convolution with the first, second or third
#             derivatives of a Gaussian. Higher order derivatives are not implemented
#         mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
#             How to handle borders
#         cval (double, optional) : 
#             If `mode` is constant, which constant to use (default: 0.0)
#         out (Tensor, optional) : 
#             Output array. Must have same shape as `array` as well as be C-contiguous. If `array` is an integer array, this must be a double array; otherwise, it must have the same type as `array`.

#     Returns:
#         filtered (Tensor) : Filtered version of `array`

#     Notes:
#         The multi-dimensional filter is implemented as a sequence of one-dimensional convolution filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a limited precision, the results may be imprecise because intermediate results may be stored with insufficient precision.
#     """
#     array = _as_floating_point_array(array)
#     output = _get_output(array, out, 'gaussian_filter', output=output)
#     orders = _normalize_sequence(array, order, 'gaussian_filter')
#     sigmas = _normalize_sequence(array, sigma, 'gaussian_filter')
#     output[...] = array[...]
#     noutput = None

#     for axis in range(array.ndim):
#         sigma = sigmas[axis]
#         order = orders[axis]
#         noutput = gaussian_filter1d(output, sigma, axis, order, mode, cval, noutput)
#         output, noutput = noutput, output

#     return output

# def _wavelet_array(f, inline, func):
#     f = _as_floating_point_array(f)

#     if f.ndim != 2:
#         raise ValueError('caer.convolve.%s: Only works for 2D images' % func)

#     if not inline:
#         return f.copy()

#     return f


# def haar(f, preserve_energy=True, inline=False):
#     """
#     Haar transform

#     Args:
#         f (2-D Tensor) : 
#             Input image
#         preserve_energy (bool, optional) : 
#             Whether to normalise the result so that energy is preserved (the default).
#         inline (bool, optional) : 
#             Whether to write the results to the input image. By default, a new image is returned. Integer images are always converted to floating point and copied.

#     """
#     f = _wavelet_array(f, inline, 'haar')
#     cconvolve.haar(f)
#     cconvolve.haar(f.T)
#     if preserve_energy:
#         f /= 2.0
#     return f


# _daubechies_codes = [('D%s' % ci) for ci in range(2,21,2)]
# def _daubechies_code(c):
#     try:
#         return _daubechies_codes.index(c)
#     except:
#         raise ValueError('caer.convolve: Known daubechies codes are {0}. You passed in {1}.'.format(_daubechies_codes, c))


# def daubechies(f, code, inline=False):
#     """
#     Daubechies wavelet transform
#     This function works best if the image sizes are powers of 2!

#     Args:
#         f (Tensor) : 
#             2-D image
#         code (str) : 
#             One of 'D2', 'D4', ... 'D20'
#         inline (bool, optional) : 
#             Whether to write the results to the input image. By default, a new image is returned. Integer images are always converted to floating point and copied.
#     """
#     f = _wavelet_array(f, inline, 'daubechies')
#     code = _daubechies_code(code)
#     cconvolve.daubechies(f, code)
#     cconvolve.daubechies(f.T, code)
#     return f


# def laplacian_2D(array, alpha = 0.2):
#     """
#     2D Laplacian filter.

#     Args:
#         array (Tensor) : 
#             input 2D array. If the array is an integer array, it will be converted 
#             to a double array.
#         alpha (scalar or sequence of scalars) : 
#             controls the shape of Laplacian operator. Must be 0-1. A larger values 
#             makes the operator empahsize the diagonal direction.
#     Returns:
#         filtered (Tensor) : Filtered version of `array`
#     """
#     array = np.array(array, dtype=np.float)
#     if array.ndim != 2:
#         raise ValueError('caer.laplacian_2D: Only available for 2-dimensional arrays')
        
#     alpha = max(0, min(alpha,1))
#     ver_hor_weight = (1. - alpha) / (alpha + 1.)
#     diag_weight = alpha / (alpha + 1.)
#     center = -4. / (alpha + 1.)
#     weights = np.array([
#     [diag_weight, ver_hor_weight, diag_weight],
#     [ver_hor_weight, center, ver_hor_weight],
#     [diag_weight, ver_hor_weight, diag_weight]])
    
#     output = convolve(array, weights, mode='nearest')

#     return output