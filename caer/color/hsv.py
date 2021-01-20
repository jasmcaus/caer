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
import numpy as np 

from ..adorad import Tensor
from .constants import HSV2BGR, HSV2RGB
from .bgr import bgr_to_gray, bgr_to_lab, bgr_to_hls, is_bgr_image

__all__ = [
    'hsv_to_rgb',
    'hsv_to_bgr',
    'hsv_to_lab',
    'hsv_to_gray',
    'hsv_to_hls',
    'is_hsv_image'
]

def is_hsv_image(img):
    return is_bgr_image(img)


def hsv_to_rgb(img) -> Tensor:
    r"""
        Converts a HSV image to its RGB version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its RGB counterpart')

    return cv.cvtColor(img, HSV2RGB)


def hsv_to_bgr(img) -> Tensor:
    r"""
        Converts a HSV image to its BGR version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its BGR counterpart')

    return cv.cvtColor(img, HSV2BGR)


def hsv_to_gray(img) -> Tensor:
    r"""
        Converts a HSV image to its Grayscale version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its Grayscale counterpart')

    bgr = hsv_to_bgr(img)

    return bgr_to_gray(bgr)


def hsv_to_lab(img) -> Tensor:
    r"""
        Converts a HSV image to its LAB version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its LAB counterpart')

    bgr = hsv_to_bgr(img)

    return bgr_to_lab(bgr)


def hsv_to_hls(img) -> Tensor:
    r"""
        Converts a HSV image to its HLS version.

    Args:
        img (Tensor): Valid HSV image array
    
    Returns:
        HLS image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hsv_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HSV image to its HLS counterpart')

    bgr = hsv_to_bgr(img)

    return bgr_to_hls(bgr)

