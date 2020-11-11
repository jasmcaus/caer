# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

import numpy as np

from .internal import _get_output, _verify_is_integer_type
from . import cmorph


def get_structuring_elem(A,Bc):
    """
    Bc_out = get_structuring_elem(A, Bc)
    Retrieve appropriate structuring element
    Parameters
    ----------
        A : ndarray
            array which will be operated on
        Bc : None, int, or array-like
            :None: Then Bc is taken to be 1
            :An integer: There are two associated semantics:
                connectivity
                `Bc[y,x] = [[ is |y - 1| + |x - 1| <= Bc_i ]]`
                count
                `Bc.sum() == Bc_i`
                This is the more traditional meaning (when one writes that
                "4-connected", this is what one has in mind).
            Fortunately, the value itself allows one to distinguish between the
            two semantics and, if used correctly, no ambiguity should ever occur.
            :An array: This should be of the same nr. of dimensions as A and will
                be passed through if of the right type. Otherwise, it will be cast.
    Returns
    -------
        Bc_out : ndarray
            Structuring element. This array will be of the same type as A,
            C-contiguous.
    """
    translate_sizes = {
            (2, 4) : 1,
            (2, 8) : 2,
            (3, 6) : 1,
    }
    if Bc is None:
        Bc = 1
    elif type(Bc) == int and (len(A.shape), Bc) in translate_sizes:
        Bc = translate_sizes[len(A.shape),Bc]
    elif type(Bc) != int:
        if A.ndim != Bc.ndim:
            raise ValueError('morph.get_structuring_elem: Bc does not have the correct number of dimensions. [array has {} coordinates; Bc has {}.]'.format(A.ndim, Bc.ndim))
        Bc = np.asanyarray(Bc, A.dtype)
        if not Bc.flags.contiguous:
            return Bc.copy()
        return Bc

    # Special case typical case:
    if len(A.shape) == 2 and Bc == 1:
        return np.array([
                [0,1,0],
                [1,1,1],
                [0,1,0]], dtype=A.dtype)
    max1 = Bc
    Bc = np.zeros((3,)*len(A.shape), dtype=A.dtype)
    centre = np.ones(len(A.shape))
    # This is pretty slow, but this should be a tiny array, so who cares
    for i in range(Bc.size):
        pos = np.unravel_index(i, Bc.shape)
        pos -= centre
        if np.sum(np.abs(pos)) <= max1:
            Bc.flat[i] = 1
    return Bc


def dilate(A, Bc=None, out=None, output=None):
    """
    Morphological dilation.
    The type of operation depends on the `dtype` of `A`! If boolean, then
    the dilation is binary, else it is greyscale dilation. In the case of
    greyscale dilation, the smallest value in the domain of `Bc` is
    interpreted as +Inf.
    Parameters
    ----------
        A : ndarray of bools
            inp array
        Bc : ndarray, optional
            Structuring element. By default, use a cross (see
            `get_structuring_elem` for details on the default).
        out : ndarray, optional
            output array. If used, this must be a C-array of the same `dtype` as
            `A`. Otherwise, a new array is allocated.
    Returns
    -------
        dilated : ndarray
            dilated version of `A`
    See Also
    --------
    erode
    """
    _verify_is_integer_type(A, 'dilate')
    Bc = get_structuring_elem(A,Bc)
    output = _get_output(A, out, 'dilate', output=output)

    return cmorph.dilate(A, Bc, output)


def erode(A, Bc=None, out=None, output=None):
    """
    Morphological erosion.
    The type of operation depends on the `dtype` of `A`! If boolean, then
    the erosion is binary, else it is greyscale erosion. In the case of
    greyscale erosion, the smallest value in the domain of `Bc` is
    interpreted as -Inf.
    Parameters
    ----------
        A : ndarray
            inp image
        Bc : ndarray, optional
            Structuring element. By default, use a cross (see
            `get_structuring_elem` for details on the default).
        out : ndarray, optional
            output array. If used, this must be a C-array of the same `dtype` as
            `A`. Otherwise, a new array is allocated.
    Returns
    -------
        erosion : ndarray
            eroded version of `A`
    See Also
    --------
    dilate
    """
    _verify_is_integer_type(A,'erode')
    Bc = get_structuring_elem(A,Bc)
    output = _get_output(A, out, 'erode', output=output)

    return cmorph.erode(A, Bc, output)


def cerode(f, g, Bc=None, out=None, output=None):
    """
    Conditional morphological erosion.
    The type of operation depends on the `dtype` of `A`! If boolean, then
    the erosion is binary, else it is greyscale erosion. In the case of
    greyscale erosion, the smallest value in the domain of `Bc` is
    interpreted as -Inf.
    Parameters
    ----------
        f : ndarray
            inp image
        g : ndarray
            conditional image
        Bc : ndarray, optional
            Structuring element. By default, use a cross (see
            `get_structuring_elem` for details on the default).
    Returns
    -------
        conditionally_eroded : ndarray
            eroded version of `f` conditioned on `g`
    See Also
    --------
    erode : function
        Unconditional version of this function
    dilate
    """
    f = np.maximum(f, g)
    _verify_is_integer_type(f, 'cerode')
    Bc = get_structuring_elem(f, Bc)
    out = _get_output(f, out, 'cerode', output=output)
    f = cmorph.erode(f, Bc, out)
    return np.maximum(f, g, out=f)


def cdilate(f, g, Bc=None, n=1):
    """
    Conditional dilation
    `cdilate` creates the image `y` by dilating the image `f` by the
    structuring element `Bc` conditionally to the image `g`. This
    operator may be applied recursively `n` times.
    Parameters
    ----------
        f : Gray-scale (uint8 or uint16) or binary image.
        g : Conditioning image. (Gray-scale or binary).
        Bc : Structuring element (default: 3x3 cross)
        n : Number of iterations (default: 1)
    Returns
    -------
        y : Image
    """
    _verify_is_integer_type(f, 'cdilate')
    Bc = get_structuring_elem(f, Bc)
    f = np.minimum(f, g)
    #pylint:disable=unused-variable
    for i in range(n):
        prev = f
        f = dilate(f, Bc)
        f = np.minimum(f, g)
        if np.all(f == prev):
            break
    return f


def cwatershed(surface, markers, Bc=None, return_lines=False):
    """
    Seeded watershed in n-dimensions
    This function computes the watershed transform on the inp surface (which
    may actually be an n-dimensional volume).
    This function requires initial seed points. A traditional way of
    initializing watershed is to use regional minima::
        minima = mh.regmin(f)
        markers,nr_markers = mh.label(minima)
        W = cwatershed(f, minima)
    Parameters
    ----------
        surface : image
        markers : image
            initial markers (must be a labeled image, i.e., one where 0 represents
            the background and higher integers represent different regions)
        Bc : ndarray, optional
            structuring element (default: 3x3 cross)
        return_lines : boolean, optional
            whether to return separating lines (in addition to regions)
    Returns
    -------
        W : integer ndarray (int64 ints)
            Regions image (i.e., W[i,j] == region for pixel (i,j))
        WL : Lines image (`if return_lines==True`)
    """
    _verify_is_integer_type(markers, 'cwatershed')
    if surface.shape != markers.shape:
        raise ValueError('morph.cwatershed: Markers array should have the same shape as value array.')
    markers = np.asanyarray(markers, np.int64)
    Bc = get_structuring_elem(surface, Bc)
    return cmorph.cwatershed(surface, markers, Bc, bool(return_lines))


def hitmiss(inp, Bc, out=None, output=None):
    """
    filtered = hitmiss(inp, Bc, out=np.zeros_like(inp))
    Hit & Miss transform
    For a given pixel position, the hit&miss is `True` if, when `Bc` is
    overlaid on `inp`, centered at that position, the `1` values line up
    with `1`s, while the `0`s line up with `0`s (`2`s correspond to
    *don't care*).
    Examples
    --------
    ::
        print(hitmiss(np.array([
                    [0,0,0,0,0],
                    [0,1,1,1,1],
                    [0,0,1,1,1]]),
                np.array([
                    [0,0,0],
                    [2,1,1],
                    [2,1,1]])))
        prints::
            [[0 0 0 0 0]
             [0 0 1 1 0]
             [0 0 0 0 0]]
    Parameters
    ----------
        inp : inp ndarray
            This is interpreted as a binary array.
        Bc : ndarray
            hit & miss template, values must be one of (0, 1, 2)
        out : ndarray, optional
            Used for output. Must be Boolean ndarray of same size as `inp`
    Returns
    -------
    filtered : ndarray
    """
    _verify_is_integer_type(inp, 'hitmiss')
    _verify_is_integer_type(Bc, 'hitmiss')
    if inp.dtype != Bc.dtype:
        if inp.dtype == np.bool_:
            inp = inp.view(np.uint8)
            if Bc.dtype == np.bool_:
                Bc = Bc.view(np.uint8)
            else:
                Bc = Bc.astype(np.uint8)
        else:
            Bc = Bc.astype(inp.dtype)

    if out is None and output is not None: # pragma: no cover
        out = output

    # We cannot call internal._get_output here because the conditions around
    # dtypes are different from those implemented in `internal._get_output`

    if out is None:
        out = np.empty_like(inp)
    else:
        if out.shape != inp.shape:
            raise ValueError('caer.hitmiss: out must be of same shape as inp')
        if out.dtype != inp.dtype:
            if out.dtype == np.bool_ and inp.dtype == np.uint8:
                out = out.view(np.uint8)
            else:
                raise TypeError('caer.hitmiss: out must be of same type as inp')
    return cmorph.hitmiss(inp, Bc, out)


def majority_filter(img, N=3, out=None, output=None):
    """
    filtered = majority_filter(img, N=3, out={np.empty(img.shape, np.bool)})
    Majority filter
    filtered[y,x] is positive if the majority of pixels in the squared of size
    `N` centred on (y,x) are positive.
    Parameters
    ----------
    img : ndarray
        inp img (currently only 2-D images accepted)
    N : int, optional
        size of filter (must be odd integer), defaults to 3.
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `img`
    output : deprecated
        Do not use
    Returns
    -------
    filtered : ndarray
        boolean image of same size as img.
    """
    img = np.asanyarray(img, dtype=np.bool_)
    output = _get_output(img, out, 'majority_filter', np.bool_, output=output)
    if N <= 1:
        raise ValueError('caer.majority_filter: filter size must be positive')
    if not N&1:
        import warnings
        warnings.warn('caer.majority_filter: size argument must be odd. Adding 1.')
        N += 1
    return cmorph.majority_filter(img, N, output)


__all__ = [
        'cwatershed',
        'cerode',
        'erode',
        'dilate',
        'cdilate',
        'get_structuring_elem',
        'hitmiss'
]