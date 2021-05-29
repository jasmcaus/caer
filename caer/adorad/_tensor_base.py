#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


class _TensorBase:
    def __init__(self):
        self.max_width = 2
        self.is_floating_point = False
        self.sci_mode = False
        self.int_mode = True
        self.numelem = self.size
        self.foreign = True

        # self._mode = 'rgb'
        # self.mode = self._mode # '._mode' is used internally --> prevents misuse of the API
        self.cspace = 'null' # default


    def __repr__(self):
        return "<class 'caer.Tensor'>"

    def __str__(self):
        return self.__repr__()

    def height(self):
        return self.shape[0]

    def width(self):
        return self.shape[1]

    def channels(self):
        return self.shape[2]

    def cmode(self):
        return self.cspace

    def numel(self):
        return self.numelem

    def dim(self):
        return self.ndim

    def size_dim(self, dim):
        return self.shape[dim]

    def type(self):
        return self.dtype

    def clone(self):
        return self.copy()

    # Colorspace stuff
    def is_rgb(self):
        return self.cspace == 'rgb'

    def is_bgr(self):
        return self.cspace == 'bgr'

    def is_gray(self):
        return self.cspace == 'gray'

    def is_hsv(self):
        return self.cspace == 'hsv'

    def is_lab(self):
        return self.cspace == 'lab'

    def is_hls(self):
        return self.cspace == 'hls'

    def is_yuv(self):
        return self.cspace == 'yuv'

    def is_luv(self):
        return self.cspace == 'luv'

    # Foreign Tensor-stuff
    def is_foreign(self):
        return self.foreign


    def _is_valid_cspace(self):
        if (self.cspace == 'rgb') or (self.cspace == 'bgr') or (self.cspace == 'gray') or (self.cspace == 'hsv') or (self.cspace == 'hls') or (self.cspace == 'lab') or (self.cspace == 'yuv') or (self.cspace == 'luv'):
            return True

        # Else
        self.cspace = 'null'
        return False


    def is_null(self):
        r"""
            Returns True if the ``.cspace`` attribute is valid (either bgr/rgb/gray/hsv/hls/lab)
            Returns False, otherwise (usually happens when foreign arrays (like ndarrays) are converted to Caer Tensors).
        """

        return not self._is_valid_cspace()


    def _nullprt(self):
        r"""
            NOT nullptr in C/C++.
            Raises a TypeError ==> usually happens when foreign arrays (like ndarrays) are converted to Caer Tensors.
        """
        if self.is_null():
            raise TypeError('IllegalTensorWarning: Cannot determine the colorspace for this foreign tensor. You can set it manually by modifying the `.cspace` attribute. We suggest operating solely in Caer Tensors.')
