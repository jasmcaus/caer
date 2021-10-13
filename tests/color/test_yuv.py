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


def test_yuv2rgb():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    rgb = caer.yuv2rgb(cv_yuv)

    assert len(rgb.shape) == 3
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_yuv2bgr():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    bgr = caer.yuv2bgr(cv_yuv)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_yuv2gray():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    gray = caer.yuv2gray(cv_yuv)

    assert len(gray.shape) == 2 or (len(gray.shape) == 3 and gray.shape[-1] == 1)
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_yuv2hsv():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    hsv = caer.yuv2hsv(cv_yuv)

    assert len(hsv.shape) == 3
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_yuv2hls():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    hls = caer.yuv2hls(cv_yuv)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_yuv2lab():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    lab = caer.yuv2lab(cv_yuv)

    assert len(lab.shape) == 3
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()


def test_yuv2luv():
    cv_yuv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2YUV)
    cv_yuv = caer.to_tensor(cv_yuv, cspace="yuv")

    luv = caer.yuv2luv(cv_yuv)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()
