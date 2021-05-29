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
from ._constants import LAB2BGR, LAB2RGB
from ._bgr import bgr2gray, bgr2hsv, bgr2hls, bgr2yuv, bgr2luv

__all__ = ['lab2rgb', 'lab2bgr', 'lab2gray', 'lab2hsv', 'lab2hls', 'lab2yuv', 'lab2luv']


def _is_lab_image(tens):
    # tens = to_tensor(tens)
    # return tens.is_lab()
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def lab2rgb(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its RGB version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        RGB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its RGB counterpart'
        )

    im = cv.cvtColor(tens, LAB2RGB)
    return to_tensor(im, cspace='rgb')


def lab2bgr(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its BGR version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its BGR counterpart'
        )

    im = cv.cvtColor(tens, LAB2BGR)
    return to_tensor(im, cspace='bgr')


def lab2gray(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its Grayscale version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its Grayscale counterpart'
        )

    bgr = lab2bgr(tens)

    im = bgr2gray(bgr)
    return to_tensor(im, cspace='gray')


def lab2hsv(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its HSV version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        HSV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its HSV counterpart'
        )

    bgr = lab2bgr(tens)

    im = bgr2hsv(bgr)
    return to_tensor(im, cspace='hsv')


def lab2hls(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its HLS version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        HLS Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its LAB counterpart'
        )

    bgr = lab2bgr(tens)

    im = bgr2hls(bgr)
    return to_tensor(im, cspace='hls')


def lab2yuv(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its YUV version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its YUV counterpart'
        )

    bgr = lab2bgr(tens)
    im = bgr2yuv(bgr)
    return to_tensor(im, cspace='yuv')


def lab2luv(tens) -> Tensor:
    r'''
        Converts an LAB Tensor to its LUV version.

    Args:
        tens (Tensor): Valid LAB Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    '''
    if not _is_lab_image(tens):
        raise ValueError(
            f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a LAB Tensor to its LUV counterpart'
        )

    bgr = lab2bgr(tens)
    im = bgr2luv(bgr)
    return to_tensor(im, cspace='luv')
