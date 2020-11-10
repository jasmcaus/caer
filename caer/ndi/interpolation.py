
# #pylint:disable=redefined-outer-name, dangerous-default-value

# import itertools
# import warnings

# import numpy as np
# from numpy.core.multiarray import normalize_axis_index

# from . import cndsupport
# from . import cndi


# __all__ = ['spline_filter1d', 'spline_filter', 'geometric_transform', 'affine_transform', 'shift', 'zoom', 'rotate']


# def spline_filter1d(inp, order=3, axis=-1, output=np.float64,
#                     mode='mirror'):
#     """
#     Calculate a 1-D spline filter along the given axis.

#     The lines of the array along the given axis are filtered by a
#     spline filter. The order of the spline must be >= 2 and <= 5.

#     Parameters
#     ----------
#     %(inp)s
#     order : int, optional
#         The order of the spline, default is 3.
#     axis : int, optional
#         The axis along which the spline filter is applied. Default is the last
#         axis.
#     output : ndarray or dtype, optional
#         The array in which to place the output, or the dtype of the returned
#         array. Default is ``np.float64``.
#     %(mode)s

#     Returns
#     -------
#     spline_filter1d : ndarray
#         The filtered inp.

#     Notes
#     -----
#     All functions in `ndimage.interpolation` do spline interpolation of
#     the inp image. If using B-splines of `order > 1`, the inp image
#     values have to be converted to B-spline coefficients first, which is
#     done by applying this 1-D filter sequentially along all
#     axes of the inp. All functions that require B-spline coefficients
#     will automatically filter their inps, a behavior controllable with
#     the `prefilter` keyword argument. For functions that accept a `mode`
#     parameter, the result will only be correct if it matches the `mode`
#     used when filtering.

#     See Also
#     --------
#     spline_filter : Multidimensional spline filter.
#     """
#     if order < 0 or order > 5:
#         raise RuntimeError('spline order not supported')
#     inp = np.asarray(inp)
#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')
#     output = cndsupport._get_output(output, inp)
#     if order in [0, 1]:
#         output[...] = np.array(inp)
#     else:
#         mode = cndsupport._extend_mode_to_code(mode)
#         axis = normalize_axis_index(axis, inp.ndim)
#         cndi.spline_filter1d(inp, order, axis, output, mode)
#     return output


# def spline_filter(inp, order=3, output=np.float64, mode='mirror'):
#     """
#     Multidimensional spline filter.

#     For more details, see `spline_filter1d`.

#     See Also
#     --------
#     spline_filter1d : Calculate a 1-D spline filter along the given axis.

#     Notes
#     -----
#     The multidimensional filter is implemented as a sequence of
#     1-D spline filters. The intermediate arrays are stored
#     in the same data type as the output. Therefore, for output types
#     with a limited precision, the results may be imprecise because
#     intermediate results may be stored with insufficient precision.
#     """
#     if order < 2 or order > 5:
#         raise RuntimeError('spline order not supported')
#     inp = np.asarray(inp)
#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')
#     output = cndsupport._get_output(output, inp)
#     if order not in [0, 1] and inp.ndim > 0:
#         for axis in range(inp.ndim):
#             spline_filter1d(inp, order, axis, output=output, mode=mode)
#             inp = output
#     else:
#         output[...] = inp[...]
#     return output 
    

# def geometric_transform(inp, mapping, output_shape=None,
#                         output=None, order=3,
#                         mode='constant', cval=0.0, prefilter=True,
#                         extra_arguments=(), extra_keywords={}):
#     """
#     Apply an arbitrary geometric transform.

#     The given mapping function is used to find, for each point in the
#     output, the corresponding coordinates in the inp. The value of the
#     inp at those coordinates is determined by spline interpolation of
#     the requested order.

#     Parameters
#     ----------
#     %(inp)s
#     mapping : {callable, scipy.LowLevelCallable}
#         A callable object that accepts a tuple of length equal to the output
#         array rank, and returns the corresponding inp coordinates as a tuple
#         of length equal to the inp array rank.
#     output_shape : tuple of ints, optional
#         Shape tuple.
#     %(output)s
#     order : int, optional
#         The order of the spline interpolation, default is 3.
#         The order has to be in the range 0-5.
#     %(mode)s
#     %(cval)s
#     %(prefilter)s
#     extra_arguments : tuple, optional
#         Extra arguments passed to `mapping`.
#     extra_keywords : dict, optional
#         Extra keywords passed to `mapping`.

#     Returns
#     -------
#     output : ndarray
#         The filtered inp.
#     """
#     if order < 0 or order > 5:
#         raise RuntimeError('spline order not supported')
#     inp = np.asarray(inp)
#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')
#     if output_shape is None:
#         output_shape = inp.shape
#     if inp.ndim < 1 or len(output_shape) < 1:
#         raise RuntimeError('inp and output rank must be > 0')
#     mode = cndsupport._extend_mode_to_code(mode)
#     if prefilter and order > 1:
#         filtered = spline_filter(inp, order, output=np.float64)
#     else:
#         filtered = inp
#     output = cndsupport._get_output(output, inp, shape=output_shape)
#     cndi.geometric_transform(filtered, mapping, None, None, None, output,
#                                   order, mode, cval, extra_arguments,
#                                   extra_keywords)
#     return output


# def affine_transform(inp, matrix, offset=0.0, output_shape=None,
#                      output=None, order=3,
#                      mode='constant', cval=0.0, prefilter=True):
#     """
#     Apply an affine transformation.

#     Given an output image pixel index vector ``o``, the pixel value
#     is determined from the inp image at position
#     ``np.dot(matrix, o) + offset``.

#     This does 'pull' (or 'backward') resampling, transforming the output space
#     to the inp to locate data. Affine transformations are often described in
#     the 'push' (or 'forward') direction, transforming inp to output. If you
#     have a matrix for the 'push' transformation, use its inverse
#     (:func:`np.linalg.inv`) in this function.

#     Parameters
#     ----------
#     %(inp)s
#     matrix : ndarray
#         The inverse coordinate transformation matrix, mapping output
#         coordinates to inp coordinates. If ``ndim`` is the number of
#         dimensions of ``inp``, the given matrix must have one of the
#         following shapes:

#             - ``(ndim, ndim)``: the linear transformation matrix for each
#               output coordinate.
#             - ``(ndim,)``: assume that the 2-D transformation matrix is
#               diagonal, with the diagonal specified by the given value. A more
#               efficient algorithm is then used that exploits the separability
#               of the problem.
#             - ``(ndim + 1, ndim + 1)``: assume that the transformation is
#               specified using homogeneous coordinates [1]_. In this case, any
#               value passed to ``offset`` is ignored.
#             - ``(ndim, ndim + 1)``: as above, but the bottom row of a
#               homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
#               and may be omitted.

#     offset : float or sequence, optional
#         The offset into the array where the transform is applied. If a float,
#         `offset` is the same for each axis. If a sequence, `offset` should
#         contain one value for each axis.
#     output_shape : tuple of ints, optional
#         Shape tuple.
#     %(output)s
#     order : int, optional
#         The order of the spline interpolation, default is 3.
#         The order has to be in the range 0-5.
#     %(mode)s
#     %(cval)s
#     %(prefilter)s

#     Returns
#     -------
#     affine_transform : ndarray
#         The transformed inp.

#     References
#     ----------
#     .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates
#     """
#     if order < 0 or order > 5:
#         raise RuntimeError('spline order not supported')
#     inp = np.asarray(inp)
#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')
#     if output_shape is None:
#         output_shape = inp.shape
#     if inp.ndim < 1 or len(output_shape) < 1:
#         raise RuntimeError('inp and output rank must be > 0')
#     mode = cndsupport._extend_mode_to_code(mode)
#     if prefilter and order > 1:
#         filtered = spline_filter(inp, order, output=np.float64)
#     else:
#         filtered = inp
#     output = cndsupport._get_output(output, inp,
#                                      shape=output_shape)
#     matrix = np.asarray(matrix, dtype=np.float64)
#     if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
#         raise RuntimeError('no proper affine matrix provided')
#     if (matrix.ndim == 2 and matrix.shape[1] == inp.ndim + 1 and
#             (matrix.shape[0] in [inp.ndim, inp.ndim + 1])):
#         if matrix.shape[0] == inp.ndim + 1:
#             exptd = [0] * inp.ndim + [1]
#             if not np.all(matrix[inp.ndim] == exptd):
#                 msg = ('Expected homogeneous transformation matrix with '
#                        'shape %s for image shape %s, but bottom row was '
#                        'not equal to %s' % (matrix.shape, inp.shape, exptd))
#                 raise ValueError(msg)
#         # assume inp is homogeneous coordinate transformation matrix
#         offset = matrix[:inp.ndim, inp.ndim]
#         matrix = matrix[:inp.ndim, :inp.ndim]
#     if matrix.shape[0] != inp.ndim:
#         raise RuntimeError('affine matrix has wrong number of rows')
#     if matrix.ndim == 2 and matrix.shape[1] != output.ndim:
#         raise RuntimeError('affine matrix has wrong number of columns')
#     if not matrix.flags.contiguous:
#         matrix = matrix.copy()
#     offset = cndsupport._normalize_sequence(offset, inp.ndim)
#     offset = np.asarray(offset, dtype=np.float64)
#     if offset.ndim != 1 or offset.shape[0] < 1:
#         raise RuntimeError('no proper offset provided')
#     if not offset.flags.contiguous:
#         offset = offset.copy()
#     if matrix.ndim == 1:
#         warnings.warn(
#             "The behavior of affine_transform with a 1-D "
#             "array supplied for the matrix parameter has changed in "
#             "SciPy 0.18.0."
#         )
#         cndi.zoom_shift(filtered, matrix, offset/matrix, output, order,
#                              mode, cval)
#     else:
#         cndi.geometric_transform(filtered, None, None, matrix, offset,
#                                       output, order, mode, cval, None, None)
#     return output



# def shift(inp, shift, output=None, order=3, mode='constant', cval=0.0,
#           prefilter=True):
#     """
#     Shift an array.

#     The array is shifted using spline interpolation of the requested order.
#     Points outside the boundaries of the inp are filled according to the
#     given mode.

#     Parameters
#     ----------
#     %(inp)s
#     shift : float or sequence
#         The shift along the axes. If a float, `shift` is the same for each
#         axis. If a sequence, `shift` should contain one value for each axis.
#     %(output)s
#     order : int, optional
#         The order of the spline interpolation, default is 3.
#         The order has to be in the range 0-5.
#     %(mode)s
#     %(cval)s
#     %(prefilter)s

#     Returns
#     -------
#     shift : ndarray
#         The shifted inp.

#     """
#     if order < 0 or order > 5:
#         raise RuntimeError('spline order not supported')
#     inp = np.asarray(inp)
#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')
#     if inp.ndim < 1:
#         raise RuntimeError('inp and output rank must be > 0')
#     mode = cndsupport._extend_mode_to_code(mode)
#     if prefilter and order > 1:
#         filtered = spline_filter(inp, order, output=np.float64)
#     else:
#         filtered = inp
#     output = cndsupport._get_output(output, inp)
#     shift = cndsupport._normalize_sequence(shift, inp.ndim)
#     shift = [-ii for ii in shift]
#     shift = np.asarray(shift, dtype=np.float64)
#     if not shift.flags.contiguous:
#         shift = shift.copy()
#     cndi.zoom_shift(filtered, None, shift, output, order, mode, cval)
#     return output



# def zoom(inp, zoom, output=None, order=3, mode='constant', cval=0.0,
#          prefilter=True):
#     """
#     Zoom an array.

#     The array is zoomed using spline interpolation of the requested order.

#     Parameters
#     ----------
#     %(inp)s
#     zoom : float or sequence
#         The zoom factor along the axes. If a float, `zoom` is the same for each
#         axis. If a sequence, `zoom` should contain one value for each axis.
#     %(output)s
#     order : int, optional
#         The order of the spline interpolation, default is 3.
#         The order has to be in the range 0-5.
#     %(mode)s
#     %(cval)s
#     %(prefilter)s

#     Returns
#     -------
#     zoom : ndarray
#         The zoomed inp
#     """
#     if order < 0 or order > 5:
#         raise RuntimeError('spline order not supported')
#     inp = np.asarray(inp)
#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')
#     if inp.ndim < 1:
#         raise RuntimeError('inp and output rank must be > 0')
#     mode = cndsupport._extend_mode_to_code(mode)
#     if prefilter and order > 1:
#         filtered = spline_filter(inp, order, output=np.float64)
#     else:
#         filtered = inp
#     zoom = cndsupport._normalize_sequence(zoom, inp.ndim)
#     output_shape = tuple(
#             [int(round(ii * jj)) for ii, jj in zip(inp.shape, zoom)])

#     zoom_div = np.array(output_shape, float) - 1
#     # Zooming to infinite values is unpredictable, so just choose
#     # zoom factor 1 instead
#     zoom = np.divide(np.array(inp.shape) - 1, zoom_div,
#                         out=np.ones_like(inp.shape, dtype=np.float64),
#                         where=zoom_div != 0)

#     output = cndsupport._get_output(output, inp,
#                                      shape=output_shape)
#     zoom = np.ascontiguousarray(zoom)
#     cndi.zoom_shift(filtered, zoom, None, output, order, mode, cval)
#     return output



# def rotate(inp, angle, axes=(1, 0), reshape=True, output=None, order=3,
#            mode='constant', cval=0.0, prefilter=True):
#     """
#     Rotate an array.

#     The array is rotated in the plane defined by the two axes given by the
#     `axes` parameter using spline interpolation of the requested order.

#     Parameters
#     ----------
#     %(inp)s
#     angle : float
#         The rotation angle in degrees.
#     axes : tuple of 2 ints, optional
#         The two axes that define the plane of rotation. Default is the first
#         two axes.
#     reshape : bool, optional
#         If `reshape` is true, the output shape is adapted so that the inp
#         array is contained completely in the output. Default is True.
#     %(output)s
#     order : int, optional
#         The order of the spline interpolation, default is 3.
#         The order has to be in the range 0-5.
#     %(mode)s
#     %(cval)s
#     %(prefilter)s

#     Returns
#     -------
#     rotate : ndarray
#         The rotated inp.
#     """
#     inp_arr = np.asarray(inp)
#     ndim = inp_arr.ndim

#     if ndim < 2:
#         raise ValueError('inp array should be at least 2D')

#     axes = list(axes)

#     if len(axes) != 2:
#         raise ValueError('axes should contain exactly two values')

#     if not all([float(ax).is_integer() for ax in axes]):
#         raise ValueError('axes should contain only integer values')

#     if axes[0] < 0:
#         axes[0] += ndim
#     if axes[1] < 0:
#         axes[1] += ndim
#     if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
#         raise ValueError('invalid rotation plane specified')

#     axes.sort()

#     c, s = special.cosdg(angle), special.sindg(angle)

#     rot_matrix = np.array([[c, s],
#                               [-s, c]])

#     img_shape = np.asarray(inp_arr.shape)
#     in_plane_shape = img_shape[axes]
#     if reshape:
#         # Compute transformed inp bounds
#         iy, ix = in_plane_shape
#         out_bounds = rot_matrix @ [[0, 0, iy, iy],
#                                    [0, ix, 0, ix]]
#         # Compute the shape of the transformed inp plane
#         out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)
#     else:
#         out_plane_shape = img_shape[axes]

#     out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
#     in_center = (in_plane_shape - 1) / 2
#     offset = in_center - out_center

#     output_shape = img_shape
#     output_shape[axes] = out_plane_shape
#     output_shape = tuple(output_shape)

#     output = cndsupport._get_output(output, inp_arr, shape=output_shape)

#     if ndim <= 2:
#         affine_transform(inp_arr, rot_matrix, offset, output_shape, output,
#                          order, mode, cval, prefilter)
#     else:
#         # If ndim > 2, the rotation is applied over all the planes
#         # parallel to axes
#         planes_coord = itertools.product(
#             *[[slice(None)] if ax in axes else range(img_shape[ax])
#               for ax in range(ndim)])

#         out_plane_shape = tuple(out_plane_shape)

#         for coordinates in planes_coord:
#             ia = inp_arr[coordinates]
#             oa = output[coordinates]
#             affine_transform(ia, rot_matrix, offset, out_plane_shape,
#                              oa, order, mode, cval, prefilter)

#     return output
