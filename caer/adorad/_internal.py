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
    'to_tensor'
]


def from_numpy(x, cspace, dtype=None, override_checks=False):
    r"""
        Convert a Numpy Array to a Caer Tensor. 
    
    .. warning::
        ``override_checks`` should not be used. If not handled property, a silly bug may pop up at an unexpected time.

    Args:
        x (ndarray): Array to convert.
        cspace (str): ``must`` be specified. Value should be either bgr/rgb/gray/hsv/hls/lab.
        dtype (numpy): (optional) Data Type 
        override_checks (bool): WARNING: Internal Usage Only => if not handled properly, a silly bug may pop up at an unexpected time.
    
    Returns:
        ``caer.Tensor``
    """

    if not override_checks:
        if cspace is None:
            raise ValueError('`cspace` must be specified ==> either bgr/rgb/gray/hsv/hls/lab.')

    if isinstance(x, np.ndarray):
        return Tensor(x, cspace=cspace, dtype=dtype)
    
    else:
        raise TypeError('`x` is not a Numpy Array')


def to_tensor(x, cspace=None, dtype=None, override_checks=False):
    r"""
        Convert an array to a caer Tensor.

        To convert a Numpy Array to a ``caer.Tensor``, specify the ``cspace`` attribute (either bgr/rgb/gray/hsv/hls/lab).
        If a ``caer.Tensor`` is passed, its attributes are preserved.

        If ``x`` is a ``caer.Tensor`` and ``cspace`` is specified, the colorspace of the Tensor will be updated accordingly. To prevent this, set ``cspace=None``. 

    .. warning::
        ``override_checks`` should not be used. If not handled property, a silly bug may pop up at an unexpected time.

    Args:
        x (Tensor, ndarray, PIL): Tensor/Array to convert.
        cspace (str): ``must`` be specified if you are converting a Numpy array to a ``caer.Tensor``. Value should be either bgr/rgb/gray/hsv/hls/lab.
        dtype (numpy): (optional) Data Type 
        override_checks (bool): WARNING: Internal Usage Only => if not handled properly, a silly bug may pop up at an unexpected time.
    """
    if isinstance(x, Tensor):
        if cspace is None: # if unsure of the cspace (usually in cases when converting a foreign array to a Caer tensor)
            return x 

        else:
            # Preserve
            x = Tensor(x, cspace=cspace, dtype=dtype)
            tens = _preserve_tensor_attrs(old=x)
            return tens 

    elif isinstance(x, np.ndarray):
        return from_numpy(x, cspace=cspace, override_checks=override_checks)

    # If PIL Image 
    elif 'PIL' in str(type(x)):
        # # If a PIL image is passed, we assume that Pillow is installed
        # from PIL import Image 
        if cspace is None:
            raise ValueError('`cspace` must be specified ==> either bgr/rgb/gray/hsv/hls/lab.')

        x = np.array(x)
        return Tensor(x, cspace=cspace, dtype=dtype)

    else:
        raise TypeError(f'Cannot convert class {type(x)} to a caer.Tensor. Currently, only Numpy arrays and (to a limited extend) PIL images) are supported.')


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

    new = Tensor(old, cspace=None, dtype=old.dtype)
    new.cspace = old.cspace 
    return new 


# def _convert_to_tensor_and_rename_cspace(x, to) -> Tensor:
#     r"""
#     Args:
#         x (Tensor, ndarray): Image Tensor
#         to (str): rgb/bgr/gray/hsv/hls/lab
        
#     Returns:
#         Tensor
#     """
#     return to_tensor(x, cspace=to)


# # def _assign_mode_to_tensor(self, rgb, gray, mode=None):
# #     r"""
# #         Assign proper value of self.cspace 

# #         Idea:
# #             rgb = True ==> self.cspace = 'rgb'
# #             rgb = False ==> self.cspace = 'bgr'
# #             gray = True ==> self.cspace = 'gray'

# #         WARNING:
# #             Use `mode` explicitely ONLY if you are inside a function that converts to HSV, HLS or LAB.
# #     """
# #     if not isinstance(rgb, bool):
# #         raise TypeError('`rgb` needs to be boolean')

# #     if not isinstance(gray, bool):
# #         raise TypeError('`gray` needs to be boolean')
    
# #     if mode is not None and isinstance(mode, str):
# #         self.cspace = mode 
    