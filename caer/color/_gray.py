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
from ._constants import GRAY2BGR, GRAY2RGB
from ._bgr import _bgr_to_lab, _bgr_to_hsv, _bgr_to_hls

__all__ = [
    '_gray_to_rgb',
    '_gray_to_bgr',
    '_gray_to_lab',
    '_gray_to_hsv',
    '_gray_to_hls',
    '_is_gray_image'
]


def _is_gray_image(img):
    img = to_tensor_(img)
    # return img.is_gray()
    return img.is_gray() or ((len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1))


def _gray_to_rgb(img) -> Tensor:
    r"""
        Converts a Grayscale image to its RGB version.

    Args:
        img (Tensor): Valid Grayscale image array
    
    Returns:
        RGB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its RGB counterpart')

    im = cv.cvtColor(img, GRAY2RGB)
    im = to_tensor_(im)
    return im 


def _gray_to_bgr(img) -> Tensor:
    r"""
        Converts a Grayscale image to its BGR version.

    Args:
        img (Tensor): Valid Grayscale image array
    
    Returns:
        BGR image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its BGR counterpart')

    im = cv.cvtColor(img, GRAY2BGR)
    im = to_tensor_(im)
    return im 


def _gray_to_lab(img) -> Tensor:
    r"""
        Converts a Grayscale image to its LAB version.

    Args:
        img (Tensor): Valid Grayscale image array
    
    Returns:
        LAB image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
    
    """
    if not _is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its LAB counterpart')

    bgr = _gray_to_bgr(img)

    return _bgr_to_lab(bgr)


def _gray_to_hsv(img) -> Tensor:
    r"""
        Converts a Grayscale image to its HSV version.

    Args:
        img (Tensor): Valid Grayscale image array
    
    Returns:
        HSV image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a LAB image to its HSV counterpart')

    bgr = _gray_to_bgr(img)

    return _bgr_to_hsv(bgr)


def _gray_to_hls(img) -> Tensor:
    r"""
        Converts a Grayscale image to its HLS version.

    Args:
        img (Tensor): Valid Grayscale image array
    
    Returns:
        HLS image array of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Image of shape 2 expected. Found shape {len(img.shape)}. This method converts a LAB image to its HLS counterpart')

    bgr = _gray_to_bgr(img)

    return _bgr_to_hls(bgr)