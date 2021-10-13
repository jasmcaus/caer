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


def test_lab2rgb():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)
    cv_lab = caer.to_tensor(cv_lab, cspace="lab")

    rgb = caer.lab2rgb(cv_lab)

    assert len(rgb.shape) == 3
    assert isinstance(rgb, caer.Tensor)
    assert rgb.is_rgb()


def test_lab2bgr():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)
    cv_lab = caer.to_tensor(cv_lab, cspace="lab")

    bgr = caer.lab2bgr(cv_lab)

    assert len(bgr.shape) == 3
    assert isinstance(bgr, caer.Tensor)
    assert bgr.is_bgr()


def test_lab2gray():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)
    cv_lab = caer.to_tensor(cv_lab, cspace="lab")

    gray = caer.lab2gray(cv_lab)

    assert len(gray.shape) == 2 or (len(gray.shape) == 3 and gray.shape[-1] == 1)
    assert isinstance(gray, caer.Tensor)
    assert gray.is_gray()


def test_lab2hsv():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)
    cv_lab = caer.to_tensor(cv_lab, cspace="lab")

    hsv = caer.lab2hsv(cv_lab)

    assert len(hsv.shape) == 3
    assert isinstance(hsv, caer.Tensor)
    assert hsv.is_hsv()


def test_lab2hls():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)
    cv_lab = caer.to_tensor(cv_lab, cspace="lab")

    hls = caer.lab2hls(cv_lab)

    assert len(hls.shape) == 3
    assert isinstance(hls, caer.Tensor)
    assert hls.is_hls()


def test_lab2yuv():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_lab = caer.to_tensor(cv_lab, cspace="rgb")
    yuv = caer.lab2yuv(cv_lab)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()


def test_lab2luv():
    cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
    cv_lab = caer.to_tensor(cv_lab, cspace="rgb")
    luv = caer.lab2luv(cv_lab)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()
