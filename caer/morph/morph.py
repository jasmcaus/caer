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

# from .._internal import _get_output, _verify_is_integer_type
# from . import cmorph


# __all__ = [
#     'cwatershed',
#     'cerode',
#     'erode',
#     'dilate',
#     'cdilate',
#     'hitmiss',
#     'regmax'
# ]

# def _get_structuring_elem(A, B):
#     """
#     Retrieve appropriate structuring element

#     Args:
#         A (Tensor) : array which will be operated on
#         B (None, int, or array-like) : 

#     Returns:
#         B_out (Tensor) : Structuring element. This array will be of the same type as A, C-contiguous.
#     """

#     translate_sizes = {
#             (2, 4) : 1,
#             (2, 8) : 2,
#             (3, 6) : 1,
#         }

#     if B is None:
#         B = 1

#     elif type(B) == int and (len(A.shape), B) in translate_sizes:
#         B = translate_sizes[len(A.shape),B]

#     elif type(B) != int:
#         if A.ndim != B.ndim:
#             raise ValueError('morph._get_structuring_elem: B does not have the correct number of dimensions. [array has {} coordinates; B has {}.]'.format(A.ndim, B.ndim))

#         B = np.asanyarray(B, A.dtype)

#         if not B.flags.contiguous:
#             return B.copy()

#         return B

#     # Special case typical case:
#     if len(A.shape) == 2 and B == 1:
#         return np.array([
#                 [0,1,0],
#                 [1,1,1],
#                 [0,1,0]], dtype=A.dtype)

#     max1 = B
#     B = np.zeros((3,)*len(A.shape), dtype=A.dtype)
#     centre = np.ones(len(A.shape))

#     # This is pretty slow, but this should be a tiny array, so it shouldn't really matter
#     for i in range(B.size):
#         pos = np.unravel_index(i, B.shape)
#         pos -= centre
#         if np.sum(np.abs(pos)) <= max1:
#             B.flat[i] = 1

#     return B


# def dilate(A, B=None, out=None, output=None):
#     """
#     Morphological dilation.

#     The type of operation depends on the `dtype` of `A`! If boolean, then the dilation is binary, else it is grayscale dilation. In the case of grayscale dilation, the smallest value in the domain of `B` is
#     interpreted as +Inf.

#     Args:
#         A (Tensor of bools) : inp array
#         B (Tensor, optional) : Structuring element. By default, use a cross.
#         out (Tensor, optional) : output array. If used, this must be a C-array of the same `dtype` as
#             `A`. Otherwise, a new array is allocated.

#     Returns:
#         dilated (Tensor) : dilated version of `A`
#     """

#     _verify_is_integer_type(A, 'dilate')

#     B = _get_structuring_elem(A,B)
#     output = _get_output(A, out, 'dilate', output=output)

#     return cmorph.dilate(A, B, output)


# def erode(A, B=None, out=None, output=None):
#     """
#     Morphological erosion.

#     The type of operation depends on the `dtype` of `A`! If boolean, then the erosion is binary, else it is grayscale erosion. In the case of grayscale erosion, the smallest value in the domain of `B` is
#     interpreted as -Inf.
    
#     Args:
#         A (Tensor) : inp image
#         B (Tensor, optional) : Structuring element. By default, uses a cross.
#         out (Tensor, optional) : output array. If used, this must be a C-array of the same `dtype` as
#             `A`. Otherwise, a new array is allocated.

#     Returns:
#         erosion (Tensor) : eroded version of `A`
#     """
#     _verify_is_integer_type(A,'erode')

#     B = _get_structuring_elem(A,B)
#     output = _get_output(A, out, 'erode', output=output)

#     return cmorph.erode(A, B, output)


# def cerode(f, g, B=None, out=None, output=None):
#     """
#     Conditional morphological erosion.

#     The type of operation depends on the `dtype` of `A`! If boolean, then the erosion is binary, else it is grayscale erosion. In the case of grayscale erosion, the smallest value in the domain of `B` is
#     interpreted as -Inf.

#     Args:
#         f (Tensor) : inp image
#         g (Tensor) : conditional image
#         B (Tensor, optional) : Structuring element. By default, use a cross.
            
#     Returns:
#         conditionally_eroded (Tensor) : eroded version of `f` conditioned on `g`
#     """

#     f = np.maximum(f, g)
#     _verify_is_integer_type(f, 'cerode')

#     B = _get_structuring_elem(f, B)
#     out = _get_output(f, out, 'cerode', output=output)
#     f = cmorph.erode(f, B, out)

#     return np.maximum(f, g, out=f)


# def cdilate(f, g, B=None, n=1):
#     """
#     Conditional dilation

#     `cdilate` creates the image `y` by dilating the image `f` by the structuring element `B` conditionally to the image `g`. This operator may be applied recursively `n` times.

#     Args:
#         f : Gray-scale (uint8 or uint16) or binary image.
#         g : Conditioning image. (Gray-scale or binary).
#         B : Structuring element (default: 3x3 cross)
#         n : Number of iterations (default: 1)

#     Returns:
#         y : Image
#     """

#     _verify_is_integer_type(f, 'cdilate')

#     B = _get_structuring_elem(f, B)
#     f = np.minimum(f, g)

#     #pylint:disable=unused-variable
#     for i in range(n):
#         prev = f
#         f = dilate(f, B)
#         f = np.minimum(f, g)

#         if np.all(f == prev):
#             break

#     return f


# def cwatershed(surface, markers, B=None, return_lines=False):
#     """
#     Seeded watershed in n-dimensions

#     This function computes the watershed transform on the inp surface (which
#     may actually be an n-dimensional volume).
#     This function requires initial seed points. A traditional way of
#     initializing watershed is to use regional minima::
#         minima = mh.regmin(f)
#         markers,nr_markers = mh.label(minima)
#         W = cwatershed(f, minima)

#     Args:
#         surface : image
#         markers : image
#             initial markers (must be a labeled image, i.e., one where 0 represents
#             the background and higher integers represent different regions)
#         B (Tensor, optional) : structuring element (default: 3x3 cross)
#         return_lines (boolean, optional) : whether to return separating lines (in addition to regions)

#     Returns:
#         W : integer Tensor (int64 ints)
#             Regions image (i.e., W[i,j] == region for pixel (i,j))
#         WL : Lines image (`if return_lines==True`)
#     """

#     _verify_is_integer_type(markers, 'cwatershed')

#     if surface.shape != markers.shape:
#         raise ValueError('morph.cwatershed: Markers array should have the same shape as value array.')

#     markers = np.asanyarray(markers, np.int64)
#     B = _get_structuring_elem(surface, B)

#     return cmorph.cwatershed(surface, markers, B, bool(return_lines))


# def hitmiss(inp, B, out=None, output=None):
#     """
#     Hit & Miss transform

#     For a given pixel position, the hit&miss is `True` if, when `B` is overlaid on `inp`, centered at that position, the `1` values line up with `1`s, while the `0`s line up with `0`s/

#     Args:
#         inp (Tensor) : This is interpreted as a binary array.
#         B (Tensor) : hit & miss template, values must be one of (0, 1, 2)
#         out (Tensor, optional) : Used for output. Must be Boolean Tensor of same size as `inp`

#     Returns:
#         filtered : Tensor
#     """
#     _verify_is_integer_type(inp, 'hitmiss')
#     _verify_is_integer_type(B, 'hitmiss')

#     if inp.dtype != B.dtype:
#         if inp.dtype == np.bool_:
#             inp = inp.view(np.uint8)

#             if B.dtype == np.bool_:
#                 B = B.view(np.uint8)
#             else:
#                 B = B.astype(np.uint8)

#         else:
#             B = B.astype(inp.dtype)

#     if out is None and output is not None: # pragma: no cover
#         out = output

#     # We cannot call .._internal._get_output here because the conditions around
#     # dtypes are different from those implemented in `.._internal._get_output`

#     if out is None:
#         out = np.empty_like(inp)

#     else:
#         if out.shape != inp.shape:
#             raise ValueError('caer.morph.hitmiss: out must be of same shape as inp')

#         if out.dtype != inp.dtype:
#             if out.dtype == np.bool_ and inp.dtype == np.uint8:
#                 out = out.view(np.uint8)

#             else:
#                 raise TypeError('caer.morph.hitmiss: out must be of same type as inp')

#     return cmorph.hitmiss(inp, B, out)


# def _remove_centre(Bc):
#     index = [s//2 for s in Bc.shape]
#     Bc[tuple(index)] = False
#     return Bc


# def regmax(f, Bc=None, out=None, output=None):
#     """
#     Regional maxima. 

#     This is a stricter criterion than the local maxima as
#     it takes the whole object into account and not just the neighbourhood
#     defined by ``Bc``::
#         0 0 0 0 0
#         0 0 2 0 0
#         0 0 2 0 0
#         0 0 3 0 0
#         0 0 3 0 0
#         0 0 0 0 0
#     The top 2 is a local maximum because it has the maximal value in its
#     neighbourhood, but it is not a regional maximum.

#     Args:
#         f (Tensor)
#         Bc (Tensor, optional) : structuring element
#         out (Tensor, optional) : Used for output. Must be Boolean Tensor of same size as `f`
#         output : deprecated
#             Do not use

#     Returns:
#         filtered (Tensor) : boolean image of same size as f.
#     """
#     Bc = _get_structuring_elem(f, Bc)
#     Bc = _remove_centre(Bc.copy())
#     output = _get_output(f, out, 'regmax', np.bool_, output=output)
#     return cmorph.regmin_max(f, Bc, output, False)