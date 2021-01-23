#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv 

from ..adorad import Tensor, to_tensor
from ._constants import BGR2RGB, BGR2GRAY, BGR2HSV, BGR2LAB, BGR2HLS

__all__ = [
    'bgr2gray',
    'bgr2hsv',
    'bgr2lab',
    'bgr2rgb',
    'bgr2hls'
]


def _is_bgr_image(tens):
    # tens = to_tensor(tens)
    # return tens.is_bgr()
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def bgr2rgb(tens) -> Tensor:
    r"""
        Converts a BGR Tensor to its RGB version.

    Args:
        tens (Tensor): Valid BGR Tensor
    
    Returns:
        RGB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_bgr_image(tens):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a BGR Tensor to its RGB counterpart')
    
    im = cv.cvtColor(tens, BGR2RGB)
    return to_tensor(im, cspace='rgb')


def bgr2gray(tens) -> Tensor:
    r"""
        Converts a BGR Tensor to its Grayscale version.

    Args:
        tens (Tensor): Valid BGR Tensor
    
    Returns:
        Grayscale Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `tens` is not of shape 3
        
    """
    if not _is_bgr_image(tens):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a BGR Tensor to its Grayscale counterpart')
    
    tens = to_tensor(tens)
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value

    im = cv.cvtColor(tens, BGR2GRAY)
    return to_tensor(im, cspace='gray')


def bgr2hsv(tens) -> Tensor:
    r"""
        Converts a BGR Tensor to its HSV version.

    Args:
        tens (Tensor): Valid BGR Tensor
    
    Returns:
        HSV Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `tens` is not of shape 3
        
    """
    if not _is_bgr_image(tens):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a BGR Tensor to its HSV counterpart')
    
    im = cv.cvtColor(tens, BGR2HSV)
    return to_tensor(im, cspace='hsv')


def bgr2lab(tens) -> Tensor:
    r"""
        Converts a BGR Tensor to its LAB version.

    Args:
        tens (Tensor): Valid BGR Tensor
    
    Returns:
        LAB Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `tens` is not of shape 3
        
    """
    if not _is_bgr_image(tens):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a BGR Tensor to its LAB counterpart')

    im = cv.cvtColor(tens, BGR2LAB)
    return to_tensor(im, cspace='lab')


def bgr2hls(tens) -> Tensor:
    r"""
        Converts a BGR Tensor to its HLS version.

    Args:
        tens (Tensor): Valid BGR Tensor
    
    Returns:
        HLS Tensor of shape ``(height, width, channels)``
    
    Raises:
        ValueError: If `tens` is not of shape 3
        
    """
    if not _is_bgr_image(tens):
        raise ValueError(f'Tensor of shape 3 expected. Found shape {len(tens.shape)}. This function converts a BGR Tensor to its HLS counterpart')
    
    im = cv.cvtColor(tens, BGR2HLS)
    return to_tensor(im, cspace='hls')