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
from ._constants import RGB2BGR, RGB2GRAY, RGB2HSV, RGB2LAB, RGB2HLS, RGB2YUV, RGB2LUV

__all__ = [
    'rgb2bgr',
    'rgb2gray',
    'rgb2hsv',
    'rgb2lab',
    'rgb2hls',
    'rgb2yuv',
    'rgb2luv',
]


def _is_rgb_image(tens):
    # tens = to_tensor(tens)
    # return tens.is_rgb()
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def rgb2bgr(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its BGR version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its BGR counterpart'
        )

    im = cv.cvtColor(tens, RGB2BGR)
    return to_tensor(im, cspace='bgr')


def rgb2gray(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its Grayscale version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its Grayscale counterpart'
        )

    im = cv.cvtColor(tens, RGB2GRAY)
    return to_tensor(im, cspace='gray')


def rgb2hsv(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its HSV version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        HSV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its HSV counterpart'
        )

    im = cv.cvtColor(tens, RGB2HSV)
    return to_tensor(im, cspace='hsv')


def rgb2hls(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its HLS version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        HLS Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its HLS counterpart'
        )

    im = cv.cvtColor(tens, RGB2HLS)
    return to_tensor(im, cspace='hls')


def rgb2lab(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its LAB version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its LAB counterpart'
        )

    im = cv.cvtColor(tens, RGB2LAB)
    return to_tensor(im, cspace='lab')


def rgb2yuv(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its YUV version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its YUV counterpart'
        )

    im = cv.cvtColor(tens, RGB2YUV)
    return to_tensor(im, cspace='yuv')


def rgb2luv(tens) -> Tensor:
    r'''
        Converts an RGB Tensor to its LUV version.

    Args:
        tens (Tensor): Valid RGB Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_rgb_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an RGB Tensor to its LUV counterpart'
        )

    im = cv.cvtColor(tens, RGB2LUV)
    return to_tensor(im, cspace='luv')
