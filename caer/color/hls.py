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
import numpy as np 
Tensor = np.ndarray

from .constants import HLS2BGR, HLS2RGB
from .bgr import bgr_to_gray, bgr_to_lab, is_bgr_image


__all__ = [
    'hls_to_rgb',
    'hls_to_bgr',
    'hls_to_lab',
    'hls_to_gray',
    'is_hls_image'
]

def is_hls_image(img):
    return is_bgr_image(img)


def hls_to_rgb(img) -> Tensor:
    r"""
        Converts a HLS image to its RGB version

    Args:
        img (ndarray): Valid HLS image array
    
    Returns:
        RGB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its RGB counterpart')

    return cv.cvtColor(img, HLS2RGB)


def hls_to_bgr(img) -> Tensor:
    r"""
        Converts a HLS image to its BGR version

    Args:
        img (ndarray): Valid HLS image array
    
    Returns:
        BGR Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its BGR counterpart')

    return cv.cvtColor(img, HLS2BGR)


def hls_to_gray(img) -> Tensor:
    r"""
        Converts a HLS image to its Grayscale version

    Args:
        img (ndarray): Valid HLS image array
    
    Returns:
        Grayscale Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its Grayscale counterpart')

    bgr = hls_to_bgr(img)

    return bgr_to_gray(bgr)


def hls_to_lab(img) -> Tensor:
    r"""
        Converts a HLS image to its LAB version

    Args:
        img (ndarray): Valid HLS image array
    
    Returns:
        LAB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a HLS image to its LAB counterpart')

    bgr = hls_to_bgr(img)

    return bgr_to_lab(bgr)

