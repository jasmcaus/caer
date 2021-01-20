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
from ._constants import LAB2BGR, LAB2RGB
from ._bgr import bgr2gray, bgr2hsv, bgr2hls

__all__ = [
    'lab2rgb',
    'lab2bgr',
    'lab2gray',
    'lab2hsv',
    'lab2hls'
]


def _is_lab_image(img):
    img = to_tensor_(img)
    # return img.is_lab()
    return img.is_lab() or (len(img.shape) == 3 and img.shape[-1] == 3)


def lab2rgb(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a LAB image to its RGB counterpart')

    im = cv.cvtColor(img, LAB2RGB)
    return _convert_to_tensor_and_rename_cspace(im, 'rgb')


def lab2bgr(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a LAB image to its BGR counterpart')

    im = cv.cvtColor(img, LAB2BGR)
    return _convert_to_tensor_and_rename_cspace(im, 'bgr')


def lab2gray(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a LAB image to its Grayscale counterpart')

    bgr = lab2bgr(img)

    im = bgr2gray(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'gray')


def lab2hsv(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a LAB image to its HSV counterpart')

    bgr = lab2bgr(img)

    im = bgr2hsv(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'hsv')


def lab2hls(img) -> Tensor:
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
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This function converts a LAB image to its LAB counterpart')

    bgr = lab2bgr(img)

    im = bgr2hls(bgr)
    return _convert_to_tensor_and_rename_cspace(im, 'hls')