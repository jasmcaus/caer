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
import cv2 as cv 
import os 

here = os.path.dirname(os.path.dirname(__file__))
img_path = os.path.join(here, 'data', 'green_fish.jpg')


def test_bgr2rgb():
    cv_bgr = cv.imread(img_path)

    rgb = caer.bgr2rgb(cv_bgr)

    assert len(rgb.shape) == 3 
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_bgr2gray():
    cv_bgr = cv.imread(img_path)

    gray = caer.bgr2gray(cv_bgr)

    assert len(gray.shape) == 2
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_bgr2hsv():
    cv_bgr = cv.imread(img_path)

    hsv = caer.bgr2hsv(cv_bgr)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_bgr2hls():
    cv_bgr = cv.imread(img_path)

    hls = caer.bgr2hls(cv_bgr)

    assert len(hls.shape) == 3 
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_bgr2lab():
    cv_bgr = cv.imread(img_path)

    lab = caer.bgr2lab(cv_bgr)

    assert len(lab.shape) == 3 
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()