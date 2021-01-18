#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv 

from ..adorad import Tensor, to_tensor_
from .constants import HLS2BGR, HLS2RGB
from .bgr import _bgr_to_gray, _bgr_to_lab, _bgr_to_hsv, _is_bgr_image


__all__ = [
    '_hls_to_rgb',
    '_hls_to_bgr',
    '_hls_to_lab',
    '_hls_to_gray',
    '_hls_to_hsv',
    '_is_hls_image'
]

def _is_hls_image(img):
    img = to_tensor_(img)
    return _is_bgr_image(img)


def _hls_to_rgb(img) -> Tensor:
    r"""
        Converts a HLS image to its RGB version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its RGB counterpart')

    im = cv.cvtColor(img, HLS2RGB)
    im = to_tensor_(im)
    return im 


def _hls_to_bgr(img) -> Tensor:
    r"""
        Converts a HLS image to its BGR version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its BGR counterpart')

    im = cv.cvtColor(img, HLS2BGR)
    im = to_tensor_(im)
    return im 


def _hls_to_gray(img) -> Tensor:
    r"""
        Converts a HLS image to its Grayscale version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its Grayscale counterpart')

    bgr = _hls_to_bgr(img)

    return _bgr_to_gray(bgr)


def _hls_to_lab(img) -> Tensor:
    r"""
        Converts a HLS image to its LAB version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its LAB counterpart')

    bgr = _hls_to_bgr(img)

    return _bgr_to_lab(bgr)


def _hls_to_hsv(img) -> Tensor:
    r"""
        Converts a HLS image to its HSV version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        HSV image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its LAB counterpart')

    bgr = _hls_to_bgr(img)

    return _bgr_to_hsv(bgr)

