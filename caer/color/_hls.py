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

from ..adorad import Tensor, to_tensor_
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
    img = to_tensor_(img)
    # return img.is_hls()
    return img.is_hls() or (len(img.shape) == 3 and img.shape[-1] == 3)


def hls2rgb(img) -> Tensor:
    r"""
        Converts a HLS image to its RGB version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its RGB counterpart')

    im = cv.cvtColor(img, HLS2RGB)
    im = to_tensor_(im)
    im.cspace = 'rgb'
    return im 


def hls2bgr(img) -> Tensor:
    r"""
        Converts a HLS image to its BGR version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its BGR counterpart')

    im = cv.cvtColor(img, HLS2BGR)
    im = to_tensor_(im)
    im.cspace = 'bgr'
    return im 


def hls2gray(img) -> Tensor:
    r"""
        Converts a HLS image to its Grayscale version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        Grayscale image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its Grayscale counterpart')

    bgr = hls2bgr(img)

    im = bgr2gray(bgr)
    im = to_tensor_(im)
    im.cspace = 'gray'
    return im 


def hls2lab(img) -> Tensor:
    r"""
        Converts a HLS image to its LAB version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its LAB counterpart')

    bgr = hls2bgr(img)

    im = bgr2lab(bgr)
    im = to_tensor_(im)
    im.cspace = 'lab'
    return im 


def hls2hsv(img) -> Tensor:
    r"""
        Converts a HLS image to its HSV version.

    Args:
        img (Tensor): Valid HLS image array
    
    Returns:
        HSV image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 3
        
    """
    if not _is_hls_image(img):
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a HLS image to its LAB counterpart')

    bgr = hls2bgr(img)

    im = bgr2hsv(bgr)
    im = to_tensor_(im)
    im.cspace = 'hsv'
    return im 