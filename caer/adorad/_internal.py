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


def from_numpy(x, cspace, dtype=None):
    r"""
        Convert a numpy array to a caer.Tensor.

    Args:
        x (ndarray): Array to convert.
        cspace (str): ``must`` be specified. Value should be either bgr/rgb/gray/hsv/hls/lab.
        dtype (numpy): (optional) Data Type 
    """

    return to_tensor(x, cspace=cspace, dtype=dtype)


class TensorWarning(UserWarning):
    pass 


def to_tensor(x, cspace=None, dtype=None):
    r"""
        Convert an array to a caer Tensor.
        To convert a Numpy Array to a caer.Tensor, specify the `cspace` attribute (either bgr/rgb/gray/hsv/hls/lab).
        If a caer Tensor is passed, its attributes are NOT preserved. For attributes to be preserved, use 
        ``to_tensor_()``.

        If ``x`` is a Tensor and ``cspace`` is specified, the colorspace of the Tensor will be updated accordingly. To prevent this, set ``cspace=None``. 

    Args:
        x (Tensor, ndarray, PIL): Tensor/Array to convert.
        cspace (str): ``must`` be specified if you are converting a Numpy array to a ``caer.Tensor``. Value should be either bgr/rgb/gray/hsv/hls/lab.
        dtype (numpy): (optional) Data Type 
    """

    if isinstance(x, np.ndarray):
        if cspace is None:
            raise ValueError('To convert a Numpy Array to a caer.Tensor, specify the `cspace` attribute (either bgr/rgb/gray/hsv/hls/lab)')

        return Tensor(x, cspace=cspace, dtype=dtype)

    elif isinstance(x, Tensor):
        if x.is_null():
            print('We get here')
            return Tensor(x, cspace=cspace, dtype=dtype)
        else:
            print('Plain return')
            return x 

    # If PIL Image 
    elif 'PIL' in str(type(x)):
        # # If a PIL image is passed, we assume that Pillow is installed
        # from PIL import Image 
        x = np.array(x)
        return Tensor(x, cspace=cspace, dtype=dtype)

    else:
        raise TypeError(f'Cannot convert class {type(x)} to a caer.Tensor. Currently, only Numpy arrays are supported.')


def to_tensor_(x, cspace, dtype=None):
    r"""
        Convert an array to a caer Tensor.
        To convert a Numpy Array to a caer.Tensor, specify the `cspace` attribute (either bgr/rgb/gray/hsv/hls/lab).
        If a caer Tensor is passed, its attributes are preserved.

        If ``x`` is a Tensor and ``cspace`` is specified, the colorspace of the Tensor will be updated accordingly. To prevent this, set ``cspace=None``. 

    Args:
        x (Tensor, ndarray, PIL): Tensor/Array to convert.
        cspace (str): ``must`` be specified if you are converting a Numpy array to a ``caer.Tensor``. Value should be either bgr/rgb/gray/hsv/hls/lab.
        dtype (numpy): (optional) Data Type
    
    Returns:
        Tensor
    """
    x = to_tensor(x, cspace=cspace, dtype=dtype)
    tens = _preserve_tensor_attrs(old=x, cspace=cspace)

    return tens 


# def to_tensor_(x, cspace=None, dtype=None):
#     r"""
#         Convert a Tensor/Array to a ``caer.Tensor``. 

#         Logic:
#             If ``x`` is a Numpy array, ``cspace` needs to be specified. Call Tensor class
#             If ``x`` is a Tensor, we preserve its attributes (ignore ``cspace``)
#     """
#     if isinstance(x, np.ndarray):
#         if cspace is None:
#             raise ValueError('To convert a Numpy Array to a caer.Tensor, specify the `cspace` attribute (either bgr/rgb/gray/hsv/hls/lab)')

#         return Tensor(x, cspace=cspace, dtype=dtype)


def _preserve_tensor_attrs(old, cspace):
    r"""
        Copies Tensor attributes (like self.cspace) of `old` in `new`. Both must be caer.Tensors 
    
    Args:
        old (Tensor): caer Tensor
    
    Returns:
        caer Tensor
    """
    if not isinstance(old, Tensor):
        raise TypeError('`old` needs to be a caer.Tensor.')

    new = to_tensor(old, cspace=cspace, dtype=old.dtype)
    new.cspace = old.cspace 
    return new 


def _check_if_tensor(x):
    if not isinstance(x, Tensor):
        raise TypeError('This function does not operate on foreign Tensors.')


def _convert_to_tensor_and_rename_cspace(x, to) -> Tensor:
    r"""
    Args:
        x (Tensor, ndarray): Image Tensor
        to (str): rgb/bgr/gray/hsv/hls/lab
        
    Returns:
        Tensor
    """
    return to_tensor_(x, cspace=to)


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
    