#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 The Caer Authors <http://github.com/jasmcaus>

import caer 
import numpy as np 
Tensor = caer.Tensor

def test_gray_to_rgb():
    img = caer.data.drone(gray=True)

    rgb = caer.gray_to_rgb(img)

    assert len(rgb.shape) == 3 
    assert isinstance(rgb, Tensor)


def test_gray_to_bgr():
    img = caer.data.drone(gray=True)

    bgr = caer.gray_to_bgr(img)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, Tensor)


def test_gray_to_hsv():
    img = caer.data.drone(gray=True)

    hsv = caer.gray_to_hsv(img)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, Tensor)


def test_gray_to_lab():
    img = caer.data.drone(gray=True)

    lab = caer.gray_to_lab(img)

    assert len(lab.shape) == 3 
    assert isinstance(lab, Tensor)