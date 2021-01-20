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

from ..adorad import Tensor, to_tensor_
from ._constants import HSV2BGR, HSV2RGB
from ._bgr import _bgr_to_gray, _bgr_to_lab, _bgr_to_hls

__all__ = [
    '_hsv_to_rgb',
    '_hsv_to_bgr',
    '_hsv_to_lab',
    '_hsv_to_gray',
    '_hsv_to_hls',
    '_is_hsv_image'
]

def _is_hsv_image(img):
    img = to_tensor_(img)
    # return img.is_hsv()
    return img.is_hsv() or (len(img.shape) == 3 and img.shape[-1] == 3)


def _hsv_to_rgb(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its RGB counterpart')

    im = cv.cvtColor(img, HSV2RGB)
    im = to_tensor_(im)
    return im 


def _hsv_to_bgr(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its BGR counterpart')

    im = cv.cvtColor(img, HSV2BGR)
    im = to_tensor_(im)
    return im 


def _hsv_to_gray(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its Grayscale counterpart')

    bgr = _hsv_to_bgr(img)

    return _bgr_to_gray(bgr)


def _hsv_to_lab(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its LAB counterpart')

    bgr = _hsv_to_bgr(img)

    return _bgr_to_lab(bgr)


def _hsv_to_hls(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its HLS counterpart')

    bgr = _hsv_to_bgr(img)

    return _bgr_to_hls(bgr)

