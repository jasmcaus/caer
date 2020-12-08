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
import cv2 as cv 
import caer 


def test_flips():
    img = caer.data.sunrise()

    vflip = caer.transforms.vflip(img)
    hflip = caer.transforms.hflip(img)
    hvflip = caer.transforms.hvflip(img)

    cv_vflip = cv.flip(img, 0)
    cv_hflip = cv.flip(img, 1)
    cv_hvflip = cv.flip(img, -1)

    # Assert same shapes
    assert vflip.shape = img.shape
    assert hflip.shape = img.shape
    assert hvflip.shape = img.shape

    # Assert everything else
    assert np.all(vflip == cv_vflip)
    assert np.all(hflip == cv_hflip)
    assert np.all(hvflip == cv_hvflip)


