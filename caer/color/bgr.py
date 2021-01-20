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

from ..adorad import Tensor
from .constants import BGR2RGB, BGR2GRAY, BGR2HSV, BGR2LAB, BGR2HLS
from .rgb import is_rgb_image

__all__ = [
    'bgr_to_gray',
    'bgr_to_hsv',
    'bgr_to_lab',
    'bgr_to_rgb',
    'bgr_to_hls',
    'is_bgr_image'
]


def is_bgr_image(img):
    return is_rgb_image(img)


def bgr_to_rgb(img) -> Tensor:
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


def bgr_to_gray(img) -> Tensor:
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


def bgr_to_hsv(img) -> Tensor:
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


def bgr_to_lab(img) -> Tensor:
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


def bgr_to_hls(img) -> Tensor:
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