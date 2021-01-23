#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>

from ..adorad import Tensor, to_tensor 

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

def to_rgb(tens) -> Tensor:
    r"""
        Converts any supported colorspace to RGB
    
    Args:
        tens (Tensor)
    
    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace 

    # If 'null', we assume we have a brand new Tensor
    if cspace == 'null':
        print('Warning: Caer was unable to assign a colorspace for a foreign tensor. Sticking with "rgb". This issue will be fixed in a future update.')
        
        # We assume that the tens is a BGR image
        im = bgr2rgb(tens)
        im = to_tensor(im, cspace='rgb')
        return im 

    elif cspace == 'rgb':
        return tens 
    
    elif cspace == 'bgr':
        im = bgr2rgb(tens)
        return to_tensor(im, cspace='rgb')
        
    elif cspace == 'gray':
        im = gray2rgb(tens)
        return to_tensor(im, cspace='rgb')
        
    elif cspace == 'hls':
        im = hls2rgb(tens)
        return to_tensor(im, cspace='rgb')
        
    elif cspace == 'hsv':
        im = hsv2rgb(tens)
        return to_tensor(im, cspace='rgb')
        
    elif cspace == 'lab':
        im = lab2rgb(tens)
        return to_tensor(im, cspace='rgb')
        

def to_bgr(tens) -> Tensor:
    r"""
        Converts any supported colorspace to BGR
    
    Args:
        tens (Tensor)
    
    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor 
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace 

    if cspace == 'bgr':
        return tens 
    
    elif cspace == 'gray':
        im = gray2bgr(tens)
        return to_tensor(im, cspace='bgr') 
    
    elif cspace == 'rgb':
        im = rgb2bgr(tens)
        return to_tensor(im, cspace='bgr') 
    
    elif cspace == 'hls':
        im = hls2bgr(tens)
        return to_tensor(im, cspace='bgr') 
    
    elif cspace == 'hsv':
        im = hsv2bgr(tens)
        return to_tensor(im, cspace='bgr') 
    
    elif cspace == 'lab':
        im = lab2bgr(tens)
        return to_tensor(im, cspace='bgr') 


def to_gray(tens) -> Tensor:
    r"""
        Converts any supported colorspace to grayscale
    
    Args:
        tens (Tensor)
    
    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor 
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace 

    if cspace == 'gray':
        return tens 
    
    elif cspace == 'bgr':
        im = bgr2gray(tens)
        return to_tensor(im, cspace='gray')
        
    elif cspace == 'rgb':
        im = rgb2gray(tens)
        return to_tensor(im, cspace='gray')
        
    elif cspace == 'hls':
        im = hls2gray(tens)
        return to_tensor(im, cspace='gray')
        
    elif cspace == 'hsv':
        im = hsv2gray(tens)
        return to_tensor(im, cspace='gray')
        
    elif cspace == 'lab':
        im = lab2gray(tens)
        return to_tensor(im, cspace='gray')
        

def to_hsv(tens) -> Tensor:
    r"""
        Converts any supported colorspace to hsv
    
    Args:
        tens (Tensor)
    
    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor 
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace 

    if cspace == 'hsv':
        return tens 
    
    elif cspace == 'bgr':
        im = bgr2hsv(tens)
        return to_tensor(im, cspace='hsv')
        
    elif cspace == 'rgb':
        im = rgb2hsv(tens)
        return to_tensor(im, cspace='hsv')
        
    elif cspace == 'hls':
        im = hls2hsv(tens)
        return to_tensor(im, cspace='hsv')
        
    elif cspace == 'gray':
        im = gray2hsv(tens)
        return to_tensor(im, cspace='hsv')
        
    elif cspace == 'lab':
        im = lab2hsv(tens)
        return to_tensor(im, cspace='hsv')
        

def to_hls(tens) -> Tensor:
    r"""
        Converts any supported colorspace to HLS
    
    Args:
        tens (Tensor)
    
    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor 
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace 

    if cspace == 'hls':
        return tens 
    
    elif cspace == 'bgr':
        im = bgr2hls(tens)
        return to_tensor(im, cspace='hls')
        
    elif cspace == 'rgb':
        im = rgb2hls(tens)
        return to_tensor(im, cspace='hls')
        
    elif cspace == 'hsv':
        im = hsv2hls(tens)
        return to_tensor(im, cspace='hls')
        
    elif cspace == 'gray':
        im = gray2hls(tens)
        return to_tensor(im, cspace='hls')
        
    elif cspace == 'lab':
        im = lab2hls(tens)
        return to_tensor(im, cspace='hls')
        

def to_lab(tens) -> Tensor:
    r"""
        Converts any supported colorspace to LAB
    
    Args:
        tens (Tensor)
    
    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor 
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace 

    if cspace == 'lab':
        return tens 
    
    elif cspace == 'bgr':
        im = bgr2lab(tens)
        return to_tensor(im, cspace='lab')
    elif cspace == 'rgb':
        im = rgb2lab(tens)
        return to_tensor(im, cspace='lab')
    elif cspace == 'hsv':
        im = hsv2lab(tens)
        return to_tensor(im, cspace='lab')
    elif cspace == 'gray':
        im = gray2lab(tens)
        return to_tensor(im, cspace='lab')
    elif cspace == 'hls':
        im = hls2lab(tens)
        return to_tensor(im, cspace='lab')