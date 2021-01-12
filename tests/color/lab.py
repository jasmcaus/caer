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
import cv2 as cv 
Tensor = caer.Tensor

def test_lab_to_rgb():
    img = caer.data.drone()
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    rgb = caer.lab_to_rgb(img)

    assert len(rgb.shape) == 3 
    assert isinstance(rgb, Tensor)


def test_lab_to_gray():
    img = caer.data.drone()
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    gray = caer.lab_to_gray(img)

    assert len(gray.shape) == 2
    assert isinstance(gray, Tensor)


def test_lab_to_bgr():
    img = caer.data.drone()
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    lab = caer.lab_to_lab(img)

    assert len(lab.shape) == 3 
    assert isinstance(lab, Tensor)


def test_lab_to_hsv():
    img = caer.data.drone()
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    lab = caer.lab_to_lab(img)

    assert len(lab.shape) == 3 
    assert isinstance(lab, Tensor)