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


def test_hsv2rgb():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="hsv")

    rgb = caer.hsv2rgb(cv_hsv)

    assert len(rgb.shape) == 3
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_hsv2bgr():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="hsv")

    bgr = caer.hsv2bgr(cv_hsv)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_hsv2gray():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="hsv")

    gray = caer.hsv2gray(cv_hsv)

    assert len(gray.shape) == 2 or (len(gray.shape) == 3 and gray.shape[-1] == 1)
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_hsv2hls():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="hsv")

    hls = caer.hsv2hls(cv_hsv)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_hsv2lab():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="hsv")

    lab = caer.hsv2lab(cv_hsv)

    assert len(lab.shape) == 3
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()


def test_hsv2yuv():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="rgb")
    yuv = caer.hsv2yuv(cv_hsv)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()


def test_hsv2luv():
    cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_hsv = caer.to_tensor(cv_hsv, cspace="rgb")
    luv = caer.hsv2luv(cv_hsv)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()
