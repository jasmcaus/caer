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
import caer 

from ._tensor_str import _str
from ._tensor_base import _TensorBase


class Tensor(_TensorBase):
    # This is required to get the type(Tensor) to be 'caer.Tensor'. 
    # Without this, type(Tensor) is 'caer.tensor.Tensor' which is not ideal.
    # Alternatively, we may shove this class to __init__.py, but this would, again, not be ideal
    __module__ = 'caer'

    def __repr__(self):
        # return "<class 'caerf.Tensor'>"
        return _str(self)


    def __new__(self, x, dtype=None):

        obj = np.array(x, dtype=dtype).view(Tensor)

        y = obj.shape
        # print('This is y:', y)
        # print(len(obj.shape)>1)
        if len(y) > 1:
            self.size = (y[1], y[0])

        else:
            self.size = y
        return obj 


def tensor(x, dtype=None):
    if not isinstance(x, (tuple, list, np.ndarray)):
        raise ValueError('Data needs to be (ideally) a list')

    return Tensor(x, dtype=dtype)


def is_tensor(obj):
    r"""
        Returns True if `obj` is a Caer tensor.

        Note that this function is simply doing ``isinstance(obj, Tensor)``. Using the ``isinstance`` check is better for typechecking with mypy, and more explicit - so it's recommended to use that instead of ``is_tensor``.

        For now, Caer Tensors are simply Numpy arrays.

    Args:
        obj (Object): Object to test
    """
    return isinstance(obj, Tensor)


def from_numpy(x, dtype=None):
    r"""
        Convert a numpy array to a Caer tensor.

    Args:
        x (ndarray): Array to convert.
    """
    x = np.asarray(x, dtype=dtype)
    return x.view(caer.Tensor)