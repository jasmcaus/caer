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
cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")


def test_rgb2bgr():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    bgr = caer.rgb2bgr(cv_rgb)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_rgb2gray():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    gray = caer.rgb2gray(cv_rgb)

    assert len(gray.shape) == 2 or (len(gray.shape) == 3 and gray.shape[-1] == 1)
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_rgb2hsv():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    hsv = caer.rgb2hsv(cv_rgb)

    assert len(hsv.shape) == 3
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_rgb2hls():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    hls = caer.rgb2hls(cv_rgb)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_rgb2lab():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    lab = caer.rgb2lab(cv_rgb)

    assert len(lab.shape) == 3
    assert isinstance(lab, caer.Tensor)
    assert lab.is_lab()


def test_rgb2yuv():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    yuv = caer.rgb2yuv(cv_rgb)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()


def test_rgb2luv():
    cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_rgb = caer.to_tensor(cv_rgb, cspace="rgb")
    luv = caer.rgb2luv(cv_rgb)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()
