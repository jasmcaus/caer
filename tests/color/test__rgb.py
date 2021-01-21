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
import cv2 as cv 
import os 

here = os.path.dirname(os.path.dirname(__file__))
img_path = os.path.join(here, 'data', 'green_fish.jpg')


def test_rgb2bgr():
    cv_rgb = cv.imread(img_path)
    cv_rgb = caer.to_tensor(cv_rgb, cspace='rgb')

    bgr = caer.rgb2bgr(cv_rgb)

    assert len(bgr.shape) == 3 
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_rgb2gray():
    cv_rgb = cv.imread(img_path)
    cv_rgb = caer.to_tensor(cv_rgb, cspace='rgb')

    gray = caer.rgb2gray(cv_rgb)

    assert len(gray.shape) == 2
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_rgb2hsv():
    cv_rgb = cv.imread(img_path)
    cv_rgb = caer.to_tensor(cv_rgb, cspace='rgb')

    hsv = caer.rgb2hsv(cv_rgb)

    assert len(hsv.shape) == 3 
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_rgb2hls():
    cv_rgb = cv.imread(img_path)
    cv_rgb = caer.to_tensor(cv_rgb, cspace='rgb')

    hls = caer.rgb2hls(cv_rgb)

    assert len(hls.shape) == 3 
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_rgb2lab():
    cv_rgb = cv.imread(img_path)
    cv_rgb = caer.to_tensor(cv_rgb, cspace='rgb')

    lab = caer.rgb2lab(cv_rgb)

    assert len(lab.shape) == 3 
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()