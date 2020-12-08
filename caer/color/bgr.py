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
from .constants import BGR2RGB, BGR2GRAY, BGR2HSV, BGR2LAB
from .rgb import is_rgb_image
import numpy as np 

__all__ = [
    'bgr_to_gray',
    'bgr_to_hsv',
    'bgr_to_lab',
    'bgr_to_rgb',
    'is_bgr_image'
]


def is_bgr_image(img):
    return is_rgb_image(img)


def bgr_to_rgb(img) -> np.ndarray:
    r"""
        Converts a BGR image to its RGB version

    Args:
        img (ndarray): Valid BGR image array
    
    Returns:
        RGB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3

    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its RGB counterpart')

    return cv.cvtColor(img, BGR2RGB)


def bgr_to_gray(img) -> np.ndarray:
    r"""
        Converts a BGR image to its Grayscale version

    Args:
        img (ndarray): Valid BGR image array
    
    Returns:
        Grayscale Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its Grayscale counterpart')
    
    return cv.cvtColor(img, BGR2GRAY)


def bgr_to_hsv(img) -> np.ndarray:
    r"""
        Converts a BGR image to its HSV version

    Args:
        img (ndarray): Valid BGR image array
    
    Returns:
        HSV Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its HSV counterpart')
    
    return cv.cvtColor(img, BGR2HSV)


def bgr_to_lab(img) -> np.ndarray:
    r"""
        Converts a BGR image to its LAB version

    Args:
        img (ndarray): Valid BGR image array
    
    Returns:
        LAB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its LAB counterpart')

    return cv.cvtColor(img, BGR2LAB)