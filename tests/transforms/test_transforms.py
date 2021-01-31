#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
import cv2 as cv 
import caer 


def test_flips():
    tens = caer.data.sunrise()

    vflip = caer.transforms.vflip(tens)
    hflip = caer.transforms.hflip(tens)
    hvflip = caer.transforms.hvflip(tens)

    cv_vflip = cv.flip(tens, 0)
    cv_hflip = cv.flip(tens, 1)
    cv_hvflip = cv.flip(tens, -1)

    # Assert same shapes
    assert vflip.shape == tens.shape
    assert hflip.shape == tens.shape
    assert hvflip.shape == tens.shape

    # Assert everything else
    assert np.all(vflip == cv_vflip)
    assert np.all(hflip == cv_hflip)
    assert np.all(hvflip == cv_hvflip)


