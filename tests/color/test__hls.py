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


def test_hls2rgb():
    cv_hls = cv.imread(img_path)

    rgb = caer.hls2rgb(cv_hls)

    assert len(rgb.shape) == 3 
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_hls2bgr():
    cv_hls = cv.imread(img_path)

    bgr = caer.hls2bgr(cv_hls)

    assert len(bgr.shape) == 3 
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_hls2gray():
    cv_hls = cv.imread(img_path)

    gray = caer.hls2gray(cv_hls)

    assert len(gray.shape) == 2
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_hls2hsv():
    cv_hls = cv.imread(img_path)

    hsv = caer.hls2hsv(cv_hls)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_hls2lab():
    cv_hls = cv.imread(img_path)

    lab = caer.hls2lab(cv_hls)

    assert len(lab.shape) == 3 
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()