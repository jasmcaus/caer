#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import cv2 as cv 
import numpy as np 

from ..adorad import Tensor
from .position import _proc_in_chunks


__all__ = [
    'blur',
    'gaussian_blur',
    'median_blur',
    'motion_blur'
]

def blur(tens, ksize) -> Tensor:
    func = _proc_in_chunks(cv.blur, ksize=(ksize, ksize))
    return func(tens)


def gaussian_blur(tens, ksize, sigma=0) -> Tensor:
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    func = _proc_in_chunks(cv.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)
    return func(tens)


def median_blur(tens, ksize) -> Tensor:
    if tens.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(
            "Invalid ksize value {}. For a float32 image the only valid ksize values are 3 and 5".format(ksize)
        )

    func = _proc_in_chunks(cv.medianBlur, ksize=ksize)
    return func(tens)


def motion_blur(tens, kernel) -> Tensor:
    func = _proc_in_chunks(cv.filter2D, ddepth=-1, kernel=kernel)
    return func(tens)