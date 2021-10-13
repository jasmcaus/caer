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


def test_bgr2rgb():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    rgb = caer.bgr2rgb(cv_bgr)

    assert len(rgb.shape) == 3
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_bgr2gray():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    gray = caer.bgr2gray(cv_bgr)

    assert len(gray.shape) == 2
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_bgr2hsv():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    hsv = caer.bgr2hsv(cv_bgr)

    assert len(hsv.shape) == 3
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_bgr2hls():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    hls = caer.bgr2hls(cv_bgr)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_bgr2lab():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    lab = caer.bgr2lab(cv_bgr)

    assert len(lab.shape) == 3
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()


def test_bgr2yuv():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    yuv = caer.bgr2yuv(cv_bgr)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()


def test_bgr2luv():
    cv_bgr = cv.imread(tens_path)
    cv_bgr = caer.to_tensor(cv_bgr, cspace="bgr")
    luv = caer.bgr2luv(cv_bgr)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()
