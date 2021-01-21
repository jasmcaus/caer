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

from ..adorad import Tensor, to_tensor_, _convert_to_tensor_and_rename_cspace
from ._constants import HLS2BGR, HLS2RGB
from ._bgr import bgr2gray, bgr2lab, bgr2hsv


__all__ = [
    'hls2rgb',
    'hls2bgr',
    'hls2lab',
    'hls2gray',
    'hls2hsv'
]

def _is_hls_image(img):
    # img = to_tensor_(img)
    # return img.is_hls()
    return len(img.shape) == 3 and img.shape[-1] == 3


def hls2rgb(img) -> Tensor:
    r"""
        Converts a HLS image to its RGB version.

    Args:
        img (Tensor): Valid HLS Tensor
    
    Returns:
        RGB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its RGB counterpart')

    im = cv.cvtColor(img, HLS2RGB)
    return _convert_to_tensor_and_rename_cspace(im, 'rgb') 


def hls2bgr(img) -> Tensor:
    r"""
        Converts a HLS image to its BGR version.

    Args:
        img (Tensor): Valid HLS Tensor
    
    Returns:
        BGR Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its BGR counterpart')

    im = cv.cvtColor(img, HLS2BGR)
    return _convert_to_tensor_and_rename_cspace(im, 'bgr')


def hls2gray(img) -> Tensor:
    r"""
        Converts a HLS image to its Grayscale version.

    Args:
        img (Tensor): Valid HLS Tensor
    
    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its Grayscale counterpart')

    bgr = hls2bgr(img)

    im = bgr2gray(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'gray')


def hls2hsv(img) -> Tensor:
    r"""
        Converts a HLS image to its HSV version.

    Args:
        img (Tensor): Valid HLS Tensor
    
    Returns:
        HSV Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its LAB counterpart')

    bgr = hls2bgr(img)

    im = bgr2hsv(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'hsv')


def hls2lab(img) -> Tensor:
    r"""
        Converts a HLS image to its LAB version.

    Args:
        img (Tensor): Valid HLS Tensor
    
    Returns:
        LAB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its LAB counterpart')

    bgr = hls2bgr(img)

    im = bgr2lab(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'lab')