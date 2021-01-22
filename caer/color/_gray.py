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
from ._constants import GRAY2BGR, GRAY2RGB
from ._bgr import bgr2lab, bgr2hsv, bgr2hls

__all__ = [
    'gray2rgb',
    'gray2bgr',
    'gray2lab',
    'gray2hsv',
    'gray2hls'
]


def _is_gray_image(img):
    # img = to_tensor(img)
    # return img.is_gray()
    return (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1)


def gray2rgb(img) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its RGB version.

    Args:
        img (Tensor): Valid Grayscale Tensor
    
    Returns:
        RGB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Tensor of shape 2 expected. Found shape {len(img.shape)}. This function converts a Grayscale Tensor to its RGB counterpart')

    im = cv.cvtColor(img, GRAY2RGB)
    return to_tensor(im, cspace='rgb')


def gray2bgr(img) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its BGR version.

    Args:
        img (Tensor): Valid Grayscale Tensor
    
    Returns:
        BGR Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Tensor of shape 2 expected. Found shape {len(img.shape)}. This function converts a Grayscale Tensor to its BGR counterpart')

    im = cv.cvtColor(img, GRAY2BGR)
    return to_tensor(im, cspace='bgr')


def gray2hsv(img) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its HSV version.

    Args:
        img (Tensor): Valid Grayscale Tensor
    
    Returns:
        HSV Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Tensor of shape 2 expected. Found shape {len(img.shape)}. This function converts a LAB Tensor to its HSV counterpart')

    bgr = gray2bgr(img)

    im = bgr2hsv(bgr)
    return to_tensor(im, cspace='hsv')


def gray2hls(img) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its HLS version.

    Args:
        img (Tensor): Valid Grayscale Tensor
    
    Returns:
        HLS Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
        
    """
    if not _is_gray_image(img):
        raise ValueError(f'Tensor of shape 2 expected. Found shape {len(img.shape)}. This function converts a LAB Tensor to its HLS counterpart')

    bgr = gray2bgr(img)

    im = bgr2hls(bgr)
    return to_tensor(im, cspace='hls')


def gray2lab(img) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its LAB version.

    Args:
        img (Tensor): Valid Grayscale Tensor
    
    Returns:
        LAB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `img` is not of shape 2
    
    """
    if not _is_gray_image(img):
        raise ValueError(f'Tensor of shape 2 expected. Found shape {len(img.shape)}. This function converts a Grayscale Tensor to its LAB counterpart')

    bgr = gray2bgr(img)

    im = bgr2lab(bgr)
    return to_tensor(im, cspace='lab')