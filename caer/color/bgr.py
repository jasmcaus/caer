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

from ..adorad import Tensor, to_tensor
from .constants import BGR2RGB, BGR2GRAY, BGR2HSV, BGR2LAB, BGR2HLS
from .rgb import is_rgb_image

__all__ = [
    '_bgr_to_gray',
    '_bgr_to_hsv',
    '_bgr_to_lab',
    '_bgr_to_rgb',
    '_bgr_to_hls',
    'is_bgr_image'
]


def is_bgr_image(img):
    return is_rgb_image(img)


def _bgr_to_rgb(img) -> Tensor:
    r"""
        Converts a BGR image to its RGB version.

    Args:
        img (Tensor): Valid BGR image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3

    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its RGB counterpart')

    return cv.cvtColor(img, BGR2RGB)


def _bgr_to_gray(img) -> Tensor:
    r"""
        Converts a BGR image to its Grayscale version.

    Args:
        img (Tensor): Valid BGR image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its Grayscale counterpart')
    
    return cv.cvtColor(img, BGR2GRAY)


def _bgr_to_hsv(img) -> Tensor:
    r"""
        Converts a BGR image to its HSV version.

    Args:
        img (Tensor): Valid BGR image array
    
    Returns:
        HSV image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its HSV counterpart')
    
    return cv.cvtColor(img, BGR2HSV)


def _bgr_to_lab(img) -> Tensor:
    r"""
        Converts a BGR image to its LAB version.

    Args:
        img (Tensor): Valid BGR image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its LAB counterpart')

    return cv.cvtColor(img, BGR2LAB)


def _bgr_to_hls(img) -> Tensor:
    r"""
        Converts a BGR image to its HLS version.

    Args:
        img (Tensor): Valid BGR image array
    
    Returns:
        HLS image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its HLS counterpart')
    
    return cv.cvtColor(img, BGR2HLS)