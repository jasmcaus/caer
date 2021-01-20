#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv 

from ..adorad import Tensor, to_tensor_, _convert_to_tensor_and_rename_cspace
from ._constants import HSV2BGR, HSV2RGB
from ._bgr import bgr2gray, bgr2lab, bgr2hls

__all__ = [
    'hsv2rgb',
    'hsv2bgr',
    'hsv2lab',
    'hsv2gray',
    'hsv2hls'
]

def _is_hsv_image(img):
    img = to_tensor_(img)
    # return img.is_hsv()
    return img.is_hsv() or (len(img.shape) == 3 and img.shape[-1] == 3)


def hsv2rgb(img) -> Tensor:
    r"""
        Converts a HSV image to its RGB version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HSV image to its RGB counterpart')

    im = cv.cvtColor(img, HSV2RGB)
    return _convert_to_tensor_and_rename_cspace(im, 'rgb')


def hsv2bgr(img) -> Tensor:
    r"""
        Converts a HSV image to its BGR version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HSV image to its BGR counterpart')

    im = cv.cvtColor(img, HSV2BGR)
    return _convert_to_tensor_and_rename_cspace(im, 'bgr')


def hsv2gray(img) -> Tensor:
    r"""
        Converts a HSV image to its Grayscale version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HSV image to its Grayscale counterpart')

    bgr = hsv2bgr(img)

    im = bgr2gray(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'gray')


def hsv2hls(img) -> Tensor:
    r"""
        Converts a HSV image to its HLS version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        HLS image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HSV image to its HLS counterpart')

    bgr = hsv2bgr(img)

    im = bgr2hls(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'hls')


def hsv2lab(img) -> Tensor:
    r"""
        Converts a HSV image to its LAB version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HSV image to its LAB counterpart')

    bgr = hsv2bgr(img)

    im = bgr2lab(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'lab')