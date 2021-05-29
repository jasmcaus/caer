#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv

from ..adorad import Tensor, to_tensor
from ._constants import HLS2BGR, HLS2RGB
from ._bgr import bgr2gray, bgr2lab, bgr2hsv, bgr2yuv, bgr2luv


__all__ = [
    'hls2rgb',
    'hls2bgr',
    'hls2lab',
    'hls2gray',
    'hls2hsv',
    'hls2yuv',
    'hls2luv',
]


def _is_hls_image(tens):
    # tens = to_tensor(tens)
    # return tens.is_hls()
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def hls2rgb(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its RGB version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        RGB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its RGB counterpart'
        )

    im = cv.cvtColor(tens, HLS2RGB)
    return to_tensor(im, cspace='rgb')


def hls2bgr(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its BGR version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its BGR counterpart'
        )

    im = cv.cvtColor(tens, HLS2BGR)
    return to_tensor(im, cspace='bgr')


def hls2gray(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its Grayscale version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its Grayscale counterpart'
        )

    bgr = hls2bgr(tens)

    im = bgr2gray(bgr)
    return to_tensor(im, cspace='gray')


def hls2hsv(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its HSV version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        HSV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its LAB counterpart'
        )

    bgr = hls2bgr(tens)

    im = bgr2hsv(bgr)
    return to_tensor(im, cspace='hsv')


def hls2lab(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its LAB version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its LAB counterpart'
        )

    bgr = hls2bgr(tens)

    im = bgr2lab(bgr)
    return to_tensor(im, cspace='lab')


def hls2yuv(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its YUV version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its YUV counterpart'
        )

    bgr = hls2bgr(tens)

    im = bgr2yuv(bgr)
    return to_tensor(im, cspace='yuv')


def hls2luv(tens) -> Tensor:
    r'''
        Converts a HLS Tensor to its LUV version.

    Args:
        tens (Tensor): Valid HLS Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hls_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HLS Tensor to its LUV counterpart'
        )

    bgr = hls2bgr(tens)

    im = bgr2luv(bgr)
    return to_tensor(im, cspace='luv')
