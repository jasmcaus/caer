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
from ._constants import HSV2BGR, HSV2RGB
from ._bgr import bgr2gray, bgr2lab, bgr2hls, bgr2yuv, bgr2luv

__all__ = [
    'hsv2rgb',
    'hsv2bgr',
    'hsv2lab',
    'hsv2gray',
    'hsv2hls',
    'hsv2yuv',
    'hsv2luv',
]


def _is_hsv_image(tens):
    # tens = to_tensor(tens)
    # return tens.is_hsv()
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def hsv2rgb(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its RGB version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        RGB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its RGB counterpart'
        )

    im = cv.cvtColor(tens, HSV2RGB)
    return to_tensor(im, cspace='rgb')


def hsv2bgr(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its BGR version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its BGR counterpart'
        )

    im = cv.cvtColor(tens, HSV2BGR)
    return to_tensor(im, cspace='bgr')


def hsv2gray(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its Grayscale version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its Grayscale counterpart'
        )

    bgr = hsv2bgr(tens)

    im = bgr2gray(bgr)
    return to_tensor(im, cspace='gray')


def hsv2hls(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its HLS version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        HLS Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its HLS counterpart'
        )

    bgr = hsv2bgr(tens)

    im = bgr2hls(bgr)
    return to_tensor(im, cspace='hls')


def hsv2lab(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its LAB version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its LAB counterpart'
        )

    bgr = hsv2bgr(tens)

    im = bgr2lab(bgr)
    return to_tensor(im, cspace='lab')


def hsv2yuv(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its YUV version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its YUV counterpart'
        )

    bgr = hsv2bgr(tens)

    im = bgr2yuv(bgr)
    return to_tensor(im, cspace='yuv')


def hsv2luv(tens) -> Tensor:
    r'''
        Converts a HSV Tensor to its LUV version.

    Args:
        tens (Tensor): Valid HSV Tensor

    Returns:
        LUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_hsv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a HSV Tensor to its LUV counterpart'
        )

    bgr = hsv2bgr(tens)

    im = bgr2luv(bgr)
    return to_tensor(im, cspace='luv')
