#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import caer
import cv2 as cv
import os

here = os.path.dirname(os.path.dirname(__file__))
tens_path = os.path.join(here, 'data', 'green_fish.jpg')


def test_gray2rgb():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    cv_gray = caer.to_tensor(cv_gray, cspace="gray")

    rgb = caer.gray2rgb(cv_gray)

    assert len(rgb.shape) == 3
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_gray2bgr():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    cv_gray = caer.to_tensor(cv_gray, cspace="gray")

    bgr = caer.gray2bgr(cv_gray)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_gray2hsv():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    cv_gray = caer.to_tensor(cv_gray, cspace="gray")

    hsv = caer.gray2hsv(cv_gray)

    assert len(hsv.shape) == 3
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_gray2hls():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    cv_gray = caer.to_tensor(cv_gray, cspace="gray")

    hls = caer.gray2hls(cv_gray)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_gray2lab():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    cv_gray = caer.to_tensor(cv_gray, cspace="gray")

    lab = caer.gray2lab(cv_gray)

    assert len(lab.shape) == 3
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()


def test_gray2yuv():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    yuv = caer.gray2yuv(cv_gray)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()


def test_gray2luv():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    luv = caer.gray2luv(cv_gray)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()
