#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

from ..adorad import Tensor, to_tensor_ 

from ._bgr import _bgr_to_gray, _bgr_to_hsv, _bgr_to_lab, _bgr_to_rgb, _bgr_to_hls
from ._rgb import _rgb_to_gray, _rgb_to_hsv, _rgb_to_lab, _rgb_to_bgr, _rgb_to_hls
from ._gray import _gray_to_lab, _gray_to_rgb, _gray_to_hsv, _gray_to_bgr, _gray_to_hls
from ._hsv import _hsv_to_gray, _hsv_to_rgb, _hsv_to_lab, _hsv_to_bgr, _hsv_to_hls
from ._hls import _hls_to_gray, _hls_to_rgb, _hls_to_lab, _hls_to_bgr, _hls_to_hsv
from ._lab import _lab_to_gray, _lab_to_rgb, _lab_to_hsv, _lab_to_bgr, _lab_to_hls

__all__ = [
    'to_rgb',
    'to_bgr',
    'to_gray',
    'to_hsv',
    'to_hls',
    'to_lab'
]

def to_rgb(img) -> Tensor:
    r"""
        Converts any supported colorspace to RGB
    
    Args:
        img (Tensor)
    
    Returns:
        Tensor
    """
    # Convert to tensor
    img = to_tensor_(img)
    cspace = img.cspace 

    if cspace == 'rgb':
        return img 
    
    elif cspace == 'bgr':
        im = _bgr_to_rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'gray':
        im = _gray_to_rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'hls':
        im = _hls_to_rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'hsv':
        im = _hsv_to_rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'lab':
        im = _lab_to_rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 


def to_bgr(img) -> Tensor:
    r"""
        Converts any supported colorspace to BGR
    
    Args:
        img (Tensor)
    
    Returns:
        Tensor
    """
    # Convert to tensor 
    img = to_tensor_(img)
    cspace = img.cspace 

    if cspace == 'bgr':
        return img 
    
    elif cspace == 'gray':
        im = _gray_to_bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'rgb':
        im = _rgb_to_bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'hls':
        im = _hls_to_bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'hsv':
        im = _hsv_to_bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'lab':
        im = _lab_to_bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 


def to_gray(img) -> Tensor:
    r"""
        Converts any supported colorspace to grayscale
    
    Args:
        img (Tensor)
    
    Returns:
        Tensor
    """
    # Convert to tensor 
    img = to_tensor_(img)
    cspace = img.cspace 

    if cspace == 'gray':
        return img 
    
    elif cspace == 'bgr':
        im = _bgr_to_gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'rgb':
        im = _rgb_to_gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'hls':
        im = _hls_to_gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'hsv':
        im = _hsv_to_gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'lab':
        im = _lab_to_gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 


def to_hsv(img) -> Tensor:
    r"""
        Converts any supported colorspace to hsv
    
    Args:
        img (Tensor)
    
    Returns:
        Tensor
    """
    # Convert to tensor 
    img = to_tensor_(img)
    cspace = img.cspace 

    if cspace == 'hsv':
        return img 
    
    elif cspace == 'bgr':
        im = _bgr_to_hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'rgb':
        im = _rgb_to_hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'hls':
        im = _hls_to_hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'gray':
        im = _gray_to_hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'lab':
        im = _lab_to_hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 


def to_hls(img) -> Tensor:
    r"""
        Converts any supported colorspace to HLS
    
    Args:
        img (Tensor)
    
    Returns:
        Tensor
    """
    # Convert to tensor 
    img = to_tensor_(img)
    cspace = img.cspace 

    if cspace == 'hls':
        return img 
    
    elif cspace == 'bgr':
        im = _bgr_to_hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'rgb':
        im = _rgb_to_hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'hsv':
        im = _hsv_to_hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'gray':
        im = _gray_to_hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'lab':
        im = _lab_to_hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 


def to_lab(img) -> Tensor:
    r"""
        Converts any supported colorspace to LAB
    
    Args:
        img (Tensor)
    
    Returns:
        Tensor
    """
    # Convert to tensor 
    img = to_tensor_(img)
    cspace = img.cspace 

    if cspace == 'lab':
        return img 
    
    elif cspace == 'bgr':
        im = _bgr_to_lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'rgb':
        im = _rgb_to_lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'hsv':
        im = _hsv_to_lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'gray':
        im = _gray_to_lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'hls':
        im = _hls_to_lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 