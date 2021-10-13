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
# BGR
cv_bgr = cv.imread(tens_path)


def test_luv2rgb():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    rgb = caer.luv2rgb(cv_luv)

    assert len(rgb.shape) == 3
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_luv2bgr():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    bgr = caer.luv2bgr(cv_luv)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_luv2gray():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    gray = caer.luv2gray(cv_luv)

    assert len(gray.shape) == 2 or (len(gray.shape) == 3 and gray.shape[-1] == 1)
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_luv2hsv():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    hsv = caer.luv2hsv(cv_luv)

    assert len(hsv.shape) == 3
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_luv2hls():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    hls = caer.luv2hls(cv_luv)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_luv2lab():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    lab = caer.luv2lab(cv_luv)

    assert len(lab.shape) == 3
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()


def test_luv2yuv():
    cv_luv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LUV)
    cv_luv = caer.to_tensor(cv_luv, cspace="luv")

    yuv = caer.luv2yuv(cv_luv)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()
