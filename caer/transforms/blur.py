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

from ..coreten import Tensor

__all__ = [
    "blur",
    "gaussian_blur",
    "median_blur",
    "motion_blur"
]

def blur(tens: Tensor, ksize:int) -> Tensor:
    return cv.blur(tens, ksize=(ksize, ksize))


def gaussian_blur(tens: Tensor, ksize:int, sigma=0) -> Tensor:
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    return cv.GaussianBlur(tens, ksize=(ksize, ksize), sigmaX=sigma)


def median_blur(tens: Tensor, ksize:int) -> Tensor:
    if tens.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(
            "Invalid ksize value {}. For a float32 image the only valid ksize values are 3 and 5".format(ksize)
        )

    return cv.medianBlur(tens, ksize=ksize)


def motion_blur(tens: Tensor, kernel) -> Tensor:
    return cv.filter2D(tens, ddepth=-1, kernel=kernel)