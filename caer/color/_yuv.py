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
from ._constants import YUV2BGR, YUV2RGB
from ._bgr import bgr2gray, bgr2hls, bgr2hsv, bgr2lab, bgr2luv

__all__ = [
    'yuv2bgr',
    'yuv2gray',
    'yuv2hls',
    'yuv2hsv',
    'yuv2lab',
    'yuv2rgb',
    'yuv2luv',
]


def _is_yuv_image(tens):
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def yuv2bgr(tens) -> Tensor:
    r'''
        Converts an YUV Tensor to its BGR version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an YUV Tensor to its BGR counterpart'
        )

    im = cv.cvtColor(tens, YUV2BGR)
    return to_tensor(im, cspace='bgr')


def yuv2rgb(tens) -> Tensor:
    r'''
        Converts a YUV Tensor to its RGB version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        RGB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a YUV Tensor to its RGB counterpart'
        )

    im = cv.cvtColor(tens, YUV2RGB)
    return to_tensor(im, cspace='rgb')


def yuv2gray(tens) -> Tensor:
    r'''
        Converts an YUV Tensor to its GRAY version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        GRAY Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an YUV Tensor to its GRAY counterpart'
        )

    im = yuv2bgr(tens)
    im = bgr2gray(tens)
    return to_tensor(im, cspace='gray')


def yuv2hls(tens) -> Tensor:
    r'''
        Converts an YUV Tensor to its HLS version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        HLS Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an YUV Tensor to its HLS counterpart'
        )

    im = yuv2bgr(tens)
    im = bgr2hls(tens)
    return to_tensor(im, cspace='hls')


def yuv2hsv(tens) -> Tensor:
    r'''
        Converts an YUV Tensor to its HSV version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        HSV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an YUV Tensor to its HSV counterpart'
        )

    im = yuv2bgr(tens)
    im = bgr2hsv(tens)
    return to_tensor(im, cspace='hsv')


def yuv2lab(tens) -> Tensor:
    r'''
        Converts an YUV Tensor to its LAB version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an YUV Tensor to its LAB counterpart'
        )

    im = yuv2bgr(tens)
    im = bgr2lab(tens)
    return to_tensor(im, cspace='lab')


def yuv2luv(tens) -> Tensor:
    r'''
        Converts an YUV Tensor to its LUV version.

    Args:
        tens (Tensor): Valid YUV Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_yuv_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts an YUV Tensor to its LUV counterpart'
        )

    im = yuv2bgr(tens)
    im = bgr2luv(tens)
    return to_tensor(im, cspace='luv')
