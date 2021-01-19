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
import numpy as np 
import os 

here = os.path.dirname(os.path.dirname(__file__))
img_path = os.path.join(here, 'data', 'drone.jpg')
# BGR
cv_bgr = cv.imread(img_path)
# RGB
cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
# GRAY
cv_gray = cv.cvtColor(cv_bgr, cv.COLOR_BGR2GRAY)
# HSV
cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
# HLS
cv_hls = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HLS)
# LAB
cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)


def test_to_rgb():
    caer_rgb_rgb = caer.to_rgb(cv_rgb)
    caer_bgr_rgb = caer.to_rgb(cv_bgr)
    caer_gray_rgb = caer.to_rgb(cv_gray)
    caer_hsv_rgb = caer.to_rgb(cv_hsv)
    caer_hls_rgb = caer.to_rgb(cv_hls)
    caer_lab_rgb = caer.to_rgb(cv_lab)

    assert isinstance(caer_rgb, caer.Tensor)
    assert caer_rgb.shape == cv_rgb.shape 
    assert caer_rgb.cspace == 'rgb'
    assert len(caer_rgb.shape) == 3
    assert np.all(caer_rgb_rgb == caer_bgr_rgb) and np.all(caer_bgr_rgb == caer_gray_rgb) and np.all(caer_gray_rgb == caer_hsv_rgb) and np.all(caer_hsv_rgb == caer_hls_rgb) and (caer_hls_rgb == caer_lab_rgb)

def test_to_bgr():
    caer_bgr_bgr = caer.to_bgr(cv_bgr)
    caer_rgb_bgr = caer.to_bgr(cv_rgb)
    caer_gray_bgr = caer.to_bgr(cv_gray)
    caer_hsv_bgr = caer.to_bgr(cv_hsv)
    caer_hls_bgr = caer.to_bgr(cv_hls)
    caer_lab_bgr = caer.to_bgr(cv_lab)

    assert isinstance(caer_bgr, caer.Tensor)
    assert caer_bgr.shape == cv_bgr.shape 
    assert caer_bgr.cspace == 'bgr'
    assert len(caer_bgr.shape) == 3
    assert np.all(caer_bgr_bgr == caer_rgb_bgr) and np.all(caer_rgb_bgr == caer_gray_bgr) and np.all(caer_gray_bgr == caer_hsv_bgr) and np.all(caer_hsv_bgr == caer_hls_bgr) and (caer_hls_bgr == caer_lab_bgr)


def test_to_gray():
    caer_gray_gray = caer.to_gray(cv_gray)
    caer_bgr_gray = caer.to_gray(cv_bgr)
    caer_rgb_gray = caer.to_gray(cv_rgb)
    caer_hsv_gray = caer.to_gray(cv_hsv)
    caer_hls_gray = caer.to_gray(cv_hls)
    caer_lab_gray = caer.to_gray(cv_lab)

    assert isinstance(caer_gray, caer.Tensor)
    assert caer_gray.shape == cv_gray.shape 
    assert caer_gray.cspace == 'gray'
    assert len(caer_gray.shape) == 2
    assert np.all(caer_gray_gray == caer_bgr_gray) and np.all(caer_bgr_gray == caer_rgb_gray) and np.all(caer_rgb_gray == caer_hsv_gray) and np.all(caer_hsv_gray == caer_hls_gray) and (caer_hls_gray == caer_lab_gray)


def test_to_hsv():
    caer_hsv_hsv = caer.to_hsv(cv_hsv)
    caer_bgr_hsv = caer.to_hsv(cv_bgr)
    caer_gray_hsv = caer.to_hsv(cv_gray)
    caer_rgb_hsv = caer.to_hsv(cv_rgb)
    caer_hls_hsv = caer.to_hsv(cv_hls)
    caer_lab_hsv = caer.to_hsv(cv_lab)

    assert isinstance(caer_hsv, caer.Tensor)
    assert caer_hsv.shape == cv_hsv.shape 
    assert caer_hsv.cspace == 'hsv'
    assert len(caer_hsv.shape) == 3
    assert np.all(caer_hsv_hsv == caer_bgr_hsv) and np.all(caer_bgr_hsv == caer_gray_hsv) and np.all(caer_gray_hsv == caer_rgb_hsv) and np.all(caer_rgb_hsv == caer_hls_hsv) and (caer_hls_hsv == caer_lab_hsv)


def test_to_hls():
    caer_hls_hls = caer.to_hls(cv_hls)
    caer_bgr_hls = caer.to_hls(cv_bgr)
    caer_gray_hls = caer.to_hls(cv_gray)
    caer_hsv_hls = caer.to_hls(cv_hsv)
    caer_rgb_hls = caer.to_hls(cv_rgb)
    caer_lab_hls = caer.to_hls(cv_lab)

    assert isinstance(caer_hls, caer.Tensor)
    assert caer_hls.shape == cv_hls.shape 
    assert caer_hls.cspace == 'hls'
    assert len(caer_hls.shape) == 3
    assert np.all(caer_hls_hls == caer_bgr_hls) and np.all(caer_bgr_hls == caer_gray_hls) and np.all(caer_gray_hls == caer_hsv_hls) and np.all(caer_hsv_hls == caer_rgb_hls) and (caer_rgb_hls == caer_lab_hls)


def test_to_lab():
    caer_lab_lab = caer.to_lab(cv_lab)
    caer_bgr_lab = caer.to_lab(cv_bgr)
    caer_gray_lab = caer.to_lab(cv_gray)
    caer_hsv_lab = caer.to_lab(cv_hsv)
    caer_hls_lab = caer.to_lab(cv_hls)
    caer_rgb_lab = caer.to_lab(cv_rgb)

    assert isinstance(caer_lab, caer.Tensor)
    assert caer_lab.shape == cv_lab.shape 
    assert caer_lab.cspace == 'lab'
    assert len(caer_lab.shape) == 3
    assert np.all(caer_lab_lab == caer_bgr_lab) and np.all(caer_bgr_lab == caer_gray_lab) and np.all(caer_gray_lab == caer_hsv_lab) and np.all(caer_hsv_lab == caer_hls_lab) and (caer_hls_lab == caer_rgb_lab)