#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
from .tensor import Tensor 

__all__ = [
    'from_numpy',
    'to_tensor',
    'to_tensor_'
]


def from_numpy(x, dtype=None):
    r"""
        Convert a numpy array to a caer.Tensor.

    Args:
        x (ndarray): Array to convert.
    """
    return to_tensor(x, dtype=dtype)


def to_tensor(x, dtype=None):
    r"""
        Convert an array to a caer Tensor
        If a caer Tensor is passed, its attributes are NOT preserved. For attributes to be preserved, use 
        ``to_tensor_()``

    Args:
        x (ndarray, Tensor, PIL): Array to convert.
        dtype (numpy): (optional) Data Type 
    """
    if isinstance(x, np.ndarray):
        return Tensor(x, dtype=dtype)

    elif isinstance(x, Tensor) and x.dtype == dtype:
        return x 
    
    elif isinstance(x, Tensor) and x.dtype != dtype:
        return Tensor(x, dtype=dtype)

    # If PIL Image 
    elif 'PIL' in str(type(x)):
        # # If a PIL image is passed, we assume that Pillow is installed
        # from PIL import Image 
        x = np.array(x)
        return Tensor(x, dtype=dtype)

    else:
        raise TypeError(f'Cannot convert class {type(x)} to a caer Tensor. Currently, only Numpy arrays are supported.')


def to_tensor_(x, dtype=None):
    r"""
        Convert an array to a caer Tensor
        If a caer Tensor is passed, its attributes are preserved. 

    Args:
        x (ndarray, Tensor, PIL): Array to convert.
        dtype (numpy): (optional) Data Type 
    """
    x = to_tensor(x, dtype=dtype)
    tens = _preserve_tensor_attrs(old=x)

    return tens 
    

def _preserve_tensor_attrs(old):
    r"""
        Copies Tensor attributes (like self.cspace) of `old` in `new`. Both must be caer.Tensors 
    
    Args:
        old (Tensor): caer Tensor
    
    Returns:
        caer Tensor
    """
    if not isinstance(old, Tensor):
        raise TypeError('`old` needs to be a caer.Tensor.')

    new = to_tensor(old, dtype=old.dtype)
    new.cspace = old.cspace 
    return new 


def _convert_to_tensor_and_rename_cspace(x, to) -> Tensor:
    r"""
    Args:
        x (Tensor, ndarray): Image Tensor
        to (str): rgb/bgr/gray/hsv/hls/lab
        
    Returns:
        Tensor
    """
    x = to_tensor_(x)
    x.cspace = to
    return x

# def _assign_mode_to_tensor(self, rgb, gray, mode=None):
#     r"""
#         Assign proper value of self.cspace 

#         Idea:
#             rgb = True ==> self.cspace = 'rgb'
#             rgb = False ==> self.cspace = 'bgr'
#             gray = True ==> self.cspace = 'gray'

#         WARNING:
#             Use `mode` explicitely ONLY if you are inside a function that converts to HSV, HLS or LAB.
#     """
#     if not isinstance(rgb, bool):
#         raise TypeError('`rgb` needs to be boolean')

#     if not isinstance(gray, bool):
#         raise TypeError('`gray` needs to be boolean')
    
#     if mode is not None and isinstance(mode, str):
#         self.cspace = mode 
    