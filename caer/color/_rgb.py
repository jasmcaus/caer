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

from ..adorad import Tensor, to_tensor
from ._constants import RGB2BGR, RGB2GRAY, RGB2HSV, RGB2LAB, RGB2HLS

__all__ = [
    'rgb2bgr',
    'rgb2gray',
    'rgb2hsv',
    'rgb2lab',
    'rgb2hls'
]

def _is_rgb_image(img):
    # img = to_tensor(img)
    # return img.is_rgb()
    return len(img.shape) == 3 and img.shape[-1] == 3


def rgb2bgr(img) -> Tensor:
    r"""
        Converts an RGB Tensor to its BGR version.

    Args:
        img (Tensor): Valid RGB Tensor
    
    Returns:
        BGR Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(img.shape)}. This function converts an RGB Tensor to its BGR counterpart')

    im = cv.cvtColor(img, RGB2BGR)
    return to_tensor(im, cspace='bgr')


def rgb2gray(img) -> Tensor:
    r"""
        Converts an RGB Tensor to its Grayscale version.

    Args:
        img (Tensor): Valid RGB Tensor
    
    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(img.shape)}. This function converts an RGB Tensor to its Grayscale counterpart')
    
    im = cv.cvtColor(img, RGB2GRAY)
    return to_tensor(im, cspace='gray')


def rgb2hsv(img) -> Tensor:
    r"""
        Converts an RGB Tensor to its HSV version.

    Args:
        img (Tensor): Valid RGB Tensor
    
    Returns:
        HSV Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(img.shape)}. This function converts an RGB Tensor to its HSV counterpart')
    
    im = cv.cvtColor(img, RGB2HSV)
    return to_tensor(im, cspace='hsv')


def rgb2hls(img) -> Tensor:
    r"""
        Converts an RGB Tensor to its HLS version.

    Args:
        img (Tensor): Valid RGB Tensor
    
    Returns:
        HLS Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(img.shape)}. This function converts an RGB Tensor to its HLS counterpart')
    
    im = cv.cvtColor(img, RGB2HLS)
    return to_tensor(im, cspace='hls')


def rgb2lab(img) -> Tensor:
    r"""
        Converts an RGB Tensor to its LAB version.

    Args:
        img (Tensor): Valid RGB Tensor
    
    Returns:
        LAB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_rgb_image(img):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(img.shape)}. This function converts an RGB Tensor to its LAB counterpart')

    im = cv.cvtColor(img, RGB2LAB)
    return to_tensor(im, cspace='lab')