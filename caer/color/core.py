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

from ._bgr import bgr2gray, bgr2hsv, bgr2lab, bgr2rgb, bgr2hls
from ._rgb import rgb2gray, rgb2hsv, rgb2lab, rgb2bgr, rgb2hls
from ._gray import gray2lab, gray2rgb, gray2hsv, gray2bgr, gray2hls
from ._hsv import hsv2gray, hsv2rgb, hsv2lab, hsv2bgr, hsv2hls
from ._hls import hls2gray, hls2rgb, hls2lab, hls2bgr, hls2hsv
from ._lab import lab2gray, lab2rgb, lab2hsv, lab2bgr, lab2hls

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

    # If 'null', we assume we have a brand new Tensor
    if cspace == 'null':
        print('Warning: Caer was unable to assign a colorspace for a foreign tensor. Sticking with "rgb". This issue will be fixed in a future update.')
        
        # We assume that the img is a BGR image
        im = bgr2rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 

    elif cspace == 'rgb':
        return img 
    
    elif cspace == 'bgr':
        im = bgr2rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'gray':
        im = gray2rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'hls':
        im = hls2rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'hsv':
        im = hsv2rgb(img)
        im = to_tensor_(im)
        im.cspace = 'rgb'
        return im 
    
    elif cspace == 'lab':
        im = lab2rgb(img)
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
        im = gray2bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'rgb':
        im = rgb2bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'hls':
        im = hls2bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'hsv':
        im = hsv2bgr(img)
        im = to_tensor_(im)
        im.cspace = 'bgr'
        return im 
    
    elif cspace == 'lab':
        im = lab2bgr(img)
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
        im = bgr2gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'rgb':
        im = rgb2gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'hls':
        im = hls2gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'hsv':
        im = hsv2gray(img)
        im = to_tensor_(im)
        im.cspace = 'gray'
        return im 
    
    elif cspace == 'lab':
        im = lab2gray(img)
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
        im = bgr2hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'rgb':
        im = rgb2hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'hls':
        im = hls2hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'gray':
        im = gray2hsv(img)
        im = to_tensor_(im)
        im.cspace = 'hsv'
        return im 
    
    elif cspace == 'lab':
        im = lab2hsv(img)
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
        im = bgr2hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'rgb':
        im = rgb2hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'hsv':
        im = hsv2hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'gray':
        im = gray2hls(img)
        im = to_tensor_(im)
        im.cspace = 'hls'
        return im 
    
    elif cspace == 'lab':
        im = lab2hls(img)
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
        im = bgr2lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'rgb':
        im = rgb2lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'hsv':
        im = hsv2lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'gray':
        im = gray2lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 
    
    elif cspace == 'hls':
        im = hls2lab(img)
        im = to_tensor_(im)
        im.cspace = 'lab'
        return im 