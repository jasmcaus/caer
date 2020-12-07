#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

import caer 
import numpy as np 

def test_rgb_to_rgb():
    img = caer.data.drone(rgb=True)

    rgb = caer.rgb_to_bgr(img)

    assert len(bgr.shape) == 3 
    assert isinstance(bgr, np.ndarray)


def test_rgb_to_gray():
    img = caer.data.drone(rgb=True)

    gray = caer.rgb_to_gray(img)

    assert len(gray.shape) == 2
    assert isinstance(gray, np.ndarray)


def test_rgb_to_hsv():
    img = caer.data.drone(rgb=True)

    hsv = caer.rgb_to_hsv(img)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, np.ndarray)


def test_rgb_to_lab():
    img = caer.data.drone(rgb=True)

    lab = caer.rgb_to_lab(img)

    assert len(lab.shape) == 3 
    assert isinstance(lab, np.ndarray)