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

from ..adorad import Tensor
from .constants import LAB2BGR, LAB2RGB
from .bgr import _bgr_to_gray, _bgr_to_hsv, _bgr_to_hls, _is_bgr_image

__all__ = [
    '_lab_to_rgb',
    '_lab_to_bgr',
    '_lab_to_gray',
    '_lab_to_hsv',
    '_lab_to_hls',
    '_is_lab_image'
]


def _is_lab_image(img):
    return _is_bgr_image(img)


def _lab_to_rgb(img) -> Tensor:
    r"""
        Converts an LAB image to its RGB version.

    Args:
        img (Tensor): Valid LAB image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its RGB counterpart')

    return cv.cvtColor(img, LAB2RGB)


def _lab_to_bgr(img) -> Tensor:
    r"""
        Converts an LAB image to its BGR version.

    Args:
        img (Tensor): Valid LAB image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its BGR counterpart')

    return cv.cvtColor(img, LAB2BGR)


def _lab_to_gray(img) -> Tensor:
    r"""
        Converts an LAB image to its Grayscale version.

    Args:
        img (Tensor): Valid LAB image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its Grayscale counterpart')

    bgr = _lab_to_bgr(img)

    return _bgr_to_gray(bgr)


def _lab_to_hsv(img) -> Tensor:
    r"""
        Converts an LAB image to its HSV version.

    Args:
        img (Tensor): Valid LAB image array
    
    Returns:
        HSV image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its LAB counterpart')

    bgr = _lab_to_bgr(img)

    return _bgr_to_hsv(bgr)


def _lab_to_hls(img) -> Tensor:
    r"""
        Converts an LAB image to its HLS version.

    Args:
        img (Tensor): Valid LAB image array
    
    Returns:
        HLS image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_lab_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its LAB counterpart')

    bgr = _lab_to_bgr(img)

    return _bgr_to_hls(bgr)