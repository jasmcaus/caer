#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import caer 
import numpy as np 
Tensor = caer.Tensor

def test_bgr_to_rgb():
    img = caer.data.drone()

    rgb = caer.bgr_to_rgb(img)

    assert len(rgb.shape) == 3 
    assert isinstance(rgb, Tensor)


def test_bgr_to_gray():
    img = caer.data.drone()

    gray = caer.bgr_to_gray(img)

    assert len(gray.shape) == 2
    assert isinstance(gray, Tensor)


def test_bgr_to_hsv():
    img = caer.data.drone()

    hsv = caer.bgr_to_hsv(img)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, Tensor)


def test_bgr_to_lab():
    img = caer.data.drone()

    lab = caer.bgr_to_lab(img)

    assert len(lab.shape) == 3 
    assert isinstance(lab, Tensor)