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
from ._constants import BGR2RGB, BGR2GRAY, BGR2HSV, BGR2LAB, BGR2HLS

__all__ = [
    '_bgr_to_gray',
    '_bgr_to_hsv',
    '_bgr_to_lab',
    '_bgr_to_rgb',
    '_bgr_to_hls',
    '_is_bgr_image'
]


def _is_bgr_image(img):
    img = to_tensor_(img)
    # return img.is_bgr()
    return img.is_bgr() or (len(img.shape) == 3 and img.shape[-1] == 3)


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
    if not _is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its RGB counterpart')
    
    im = cv.cvtColor(img, BGR2RGB)
    im = to_tensor_(im)
    return im 


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
    if not _is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its Grayscale counterpart')
    
    im = cv.cvtColor(img, BGR2GRAY)
    im = to_tensor_(im)
    return im 


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
    if not _is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its HSV counterpart')
    
    im = cv.cvtColor(img, BGR2HSV)
    im = to_tensor_(im)
    return im 


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
    if not _is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its LAB counterpart')

    im = cv.cvtColor(img, BGR2LAB)
    im = to_tensor_(im)
    return im 


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
    if not _is_bgr_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a BGR image to its HLS counterpart')
    
    im = cv.cvtColor(img, BGR2HLS)
    im = to_tensor_(im)
    return im 