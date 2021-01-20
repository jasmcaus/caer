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


def test_gray2rgb():
    cv_gray = cv.imread(img_path)

    rgb = caer.gray2rgb(cv_gray)

    assert len(rgb.shape) == 3 
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_gray2bgr():
    cv_gray = cv.imread(img_path)

    bgr = caer.gray2bgr(cv_gray)

    assert len(bgr.shape) == 3 
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_gray2hsv():
    cv_gray = cv.imread(img_path)

    hsv = caer.gray2hsv(cv_gray)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_gray2hls():
    cv_gray = cv.imread(img_path)

    hls = caer.gray2hls(cv_gray)

    assert len(hls.shape) == 3 
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_gray2lab():
    cv_gray = cv.imread(img_path)

    lab = caer.gray2lab(cv_gray)

    assert len(lab.shape) == 3 
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()