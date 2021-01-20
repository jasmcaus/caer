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
from ._constants import RGB2BGR, RGB2GRAY, RGB2HSV, RGB2LAB, RGB2HLS

__all__ = [
    '_rgb_to_bgr',
    '_rgb_to_gray',
    '_rgb_to_hsv',
    '_rgb_to_lab',
    '_rgb_to_hls',
    '_is_rgb_image',
]

def _is_rgb_image(img):
    img = to_tensor_(img)
    return img.is_rgb()
    # return len(img.shape) == 3 and img.shape[-1] == 3


def _rgb_to_bgr(img) -> Tensor:
    r"""
        Converts an RGB image to its BGR version.

    Args:
        img (Tensor): Valid RGB image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts an RGB image to its BGR counterpart')

    im = cv.cvtColor(img, RGB2BGR)
    im = to_tensor_(im)
    return im 


def _rgb_to_gray(img) -> Tensor:
    r"""
        Converts an RGB image to its Grayscale version.

    Args:
        img (Tensor): Valid RGB image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts an RGB image to its Grayscale counterpart')
    
    im = cv.cvtColor(img, RGB2GRAY)
    im = to_tensor_(im)
    return im 


def _rgb_to_hsv(img) -> Tensor:
    r"""
        Converts an RGB image to its HSV version.

    Args:
        img (Tensor): Valid RGB image array
    
    Returns:
        HSV image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts an RGB image to its HSV counterpart')
    
    im = cv.cvtColor(img, RGB2HSV)
    im = to_tensor_(im)
    return im 


def _rgb_to_lab(img) -> Tensor:
    r"""
        Converts an RGB image to its LAB version.

    Args:
        img (Tensor): Valid RGB image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts an RGB image to its LAB counterpart')

    im = cv.cvtColor(img, RGB2LAB)
    im = to_tensor_(im)
    return im 


def _rgb_to_hls(img) -> Tensor:
    r"""
        Converts an RGB image to its HLS version.

    Args:
        img (Tensor): Valid RGB image array
    
    Returns:
        HLS image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts an RGB image to its HLS counterpart')
    
    im = cv.cvtColor(img, RGB2HLS)
    im = to_tensor_(im)
    return im 