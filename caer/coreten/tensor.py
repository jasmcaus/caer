#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

# pylint:disable=unused-argument

import numpy as np

from ._tensor_str import _str
from ._tensor_base import _TensorBase


# We use np.ndarray as a super class, because ``ndarray.view()`` expects an ndarray sub-class
# We also derive useful class methods from ``_TensorBase`` which serves as the Tensor's Base
class Tensor(_TensorBase, np.ndarray):
    # This is required to get the type(Tensor) to be 'caer.Tensor'.
    # Without this, type(Tensor) is 'caer.tensor.Tensor' which is not ideal.
    # Alternatively, we may shove this class to __init__.py, but this would, again, not be ideal
    __module__ = 'caer'

    # def __new__(cls, x, dtype=None):
    def __new__(cls, x, cspace, dtype=None):
        if not isinstance(x, (tuple, list, np.ndarray)):
            raise TypeError('Data needs to be (ideally) a list')

        if not isinstance(cspace, str) and cspace is not None:
            raise TypeError(
                f'`cspace` needs to be of type <string>, not {type(cspace)}'
            )

        obj = np.asarray(x, dtype=dtype).view(cls)
        obj.dtype = obj.dtype

        return obj

    def __init__(self, x, cspace, dtype=None):
        super().__init__()  # gets attributes from '_TensorBase'
        self.x = x

        if cspace is None:
            self.cspace = 'null'

        else:
            if cspace in ('rgb', 'bgr', 'gray', 'hsv', 'hls', 'lab', 'yuv', 'luv'):
                self.cspace = cspace
            else:
                raise ValueError(
                    'The `cspace` attribute needs to be either rgb/bgr/gray/hsv/hls/lab/yuv/luv'
                )

    def __repr__(self):
        return _str(self)

    def __str__(self):
        return self.__repr__()


def is_tensor(x):
    r'''
        Returns True if `x` is a Caer tensor.

        Note that this function is simply doing ``isinstance(x, Tensor)``. Using the ``isinstance`` check is better for typechecking with mypy, and more explicit - so it's recommended to use that instead of ``is_tensor``.

        For now, Caer Tensors are simply customized Numpy arrays.

    Args:
        x (Object): Object to test
    '''
    return 'caer.Tensor' in str(type(x)) or (isinstance(x, Tensor))
