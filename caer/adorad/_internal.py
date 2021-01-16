#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
from .tensor import Tensor 


def from_numpy(x, dtype=None):
    r"""
        Convert a numpy array to a Caer tensor.

    Args:
        x (ndarray): Array to convert.
    """
    return to_tensor(x, dtype=dtype)


def to_tensor(x, dtype=None):
    r"""
    Convert a numpy array to a Caer tensor.

    Args:
        x (ndarray): Array to convert.
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
    

# def _assign_mode_to_tensor(self, rgb, gray, mode=None):
#     r"""
#         Assign proper value of self._mode 

#         Idea:
#             rgb = True ==> self._mode = 'rgb'
#             rgb = False ==> self._mode = 'bgr'
#             gray = True ==> self._mode = 'gray'

#         WARNING:
#             Use `mode` explicitely ONLY if you are inside a function that converts to HSV, HLS or LAB.
#     """
#     if not isinstance(rgb, bool):
#         raise TypeError('`rgb` needs to be boolean')

#     if not isinstance(gray, bool):
#         raise TypeError('`gray` needs to be boolean')
    
#     if mode is not None and isinstance(mode, str):
#         self._mode = mode 
    