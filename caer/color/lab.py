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

from .constants import LAB2BGR, LAB2RGB
from .bgr import bgr_to_gray, bgr_to_hsv, is_bgr_image

__all__ = [
    'lab_to_rgb',
    'lab_to_bgr',
    'lab_to_hsv',
    'lab_to_gray',
    'is_lab_image'
]


def is_lab_image(img):
    return is_bgr_image(img)


def lab_to_rgb(img) -> np.ndarray:
    r"""
        Converts an LAB image to its RGB version

    Args:
        img (ndarray): Valid LAB image array
    
    Returns:
        RGB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its RGB counterpart')

    return cv.cvtColor(img, LAB2RGB)


def lab_to_bgr(img) -> np.ndarray:
    r"""
        Converts an LAB image to its BGR version

    Args:
        img (ndarray): Valid LAB image array
    
    Returns:
        BGR Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its BGR counterpart')

    return cv.cvtColor(img, LAB2BGR)


def lab_to_gray(img) -> np.ndarray:
    r"""
        Converts an LAB image to its Grayscale version

    Args:
        img (ndarray): Valid LAB image array
    
    Returns:
        Grayscale Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its Grayscale counterpart')

    bgr = lab_to_bgr(img)

    return bgr_to_gray(bgr)


def lab_to_hsv(img) -> np.ndarray:
    r"""
        Converts an LAB image to its HSV version

    Args:
        img (ndarray): Valid LAB image array
    
    Returns:
        HSV Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its LAB counterpart')

    bgr = lab_to_bgr(img)

    return bgr_to_hsv(bgr)

