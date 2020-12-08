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

from .constants import GRAY2BGR, GRAY2RGB
from .bgr import bgr_to_lab, bgr_to_hsv

__all__ = [
    'gray_to_rgb',
    'gray_to_bgr',
    'gray_to_hsv',
    'gray_to_lab',
    'is_gray_image'
]


def is_gray_image(img):
    return (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1)


def gray_to_rgb(img) -> np.ndarray:
    r"""
        Converts a Grayscale image to its RGB version

    Args:
        img (ndarray): Valid Grayscale image array
    
    Returns:
        RGB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its RGB counterpart')

    return cv.cvtColor(img, GRAY2RGB)


def gray_to_bgr(img) -> np.ndarray:
    r"""
        Converts a Grayscale image to its BGR version

    Args:
        img (ndarray): Valid Grayscale image array
    
    Returns:
        BGR Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its BGR counterpart')

    return cv.cvtColor(img, GRAY2BGR)


def gray_to_lab(img) -> np.ndarray:
    r"""
        Converts a Grayscale image to its LAB version

    Args:
        img (ndarray): Valid Grayscale image array
    
    Returns:
        LAB Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its LAB counterpart')

    bgr = gray_to_bgr(img)

    return bgr_to_lab(bgr)


def gray_to_hsv(img) -> np.ndarray:
    r"""
        Converts a Grayscale image to its HSV version

    Args:
        img (ndarray): Valid Grayscale image array
    
    Returns:
        HSV Image (ndarray)
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a LAB image to its HSV counterpart')

    bgr = gray_to_bgr(img)

    return bgr_to_hsv(bgr)

