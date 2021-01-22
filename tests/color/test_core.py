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

# BGR
cv_bgr = cv.imread(img_path)
cv_bgr = caer.to_tensor(cv_bgr, cspace='bgr')
# RGB
cv_rgb = cv.cvtColor(cv_bgr, cv.COLOR_BGR2RGB)
cv_rgb = caer.to_tensor(cv_rgb, cspace='rgb')
# GRAY
cv_gray = cv.cvtColor(cv_bgr, cv.COLOR_BGR2GRAY)
cv_gray = caer.to_tensor(cv_gray, cspace='gray')
# HSV
cv_hsv = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HSV)
cv_hsv = caer.to_tensor(cv_hsv, cspace='hsv')
# HLS
cv_hls = cv.cvtColor(cv_bgr, cv.COLOR_BGR2HLS)
cv_hls = caer.to_tensor(cv_hls, cspace='hls')
# LAB
cv_lab = cv.cvtColor(cv_bgr, cv.COLOR_BGR2LAB)
cv_lab = caer.to_tensor(cv_lab, cspace='lab')


def test_to_rgb():
    caer_rgb_rgb = caer.to_rgb(cv_rgb)
    caer_bgr_rgb = caer.to_rgb(cv_bgr)
    caer_gray_rgb = caer.to_rgb(cv_gray)
    caer_hsv_rgb = caer.to_rgb(cv_hsv)
    caer_hls_rgb = caer.to_rgb(cv_hls)
    caer_lab_rgb = caer.to_rgb(cv_lab)

    assert isinstance(caer_rgb_rgb, caer.Tensor)
    assert isinstance(caer_bgr_rgb, caer.Tensor)
    assert isinstance(caer_gray_rgb, caer.Tensor)
    assert isinstance(caer_hsv_rgb, caer.Tensor)
    assert isinstance(caer_hls_rgb, caer.Tensor)
    assert isinstance(caer_lab_rgb, caer.Tensor)

    assert caer_rgb_rgb.shape == cv_rgb.shape 
    assert caer_bgr_rgb.shape == cv_rgb.shape 
    assert caer_gray_rgb.shape == cv_rgb.shape 
    assert caer_hsv_rgb.shape == cv_rgb.shape 
    assert caer_hls_rgb.shape == cv_rgb.shape 
    assert caer_lab_rgb.shape == cv_rgb.shape 

    assert caer_rgb_rgb.cspace == 'rgb'
    assert caer_bgr_rgb.cspace == 'rgb'
    assert caer_gray_rgb.cspace == 'rgb'
    assert caer_hsv_rgb.cspace == 'rgb'
    assert caer_hls_rgb.cspace == 'rgb'
    assert caer_lab_rgb.cspace == 'rgb'

    assert len(caer_rgb_rgb.shape) == 3
    assert len(caer_bgr_rgb.shape) == 3
    assert len(caer_gray_rgb.shape) == 3
    assert len(caer_hsv_rgb.shape) == 3
    assert len(caer_hls_rgb.shape) == 3
    assert len(caer_lab_rgb.shape) == 3

    # assert np.all(caer_rgb_rgb == caer_bgr_rgb) 
    # assert np.all(caer_bgr_rgb == caer_hsv_rgb) 
    # assert np.all(caer_bgr_rgb == caer_gray_rgb) 
    # assert np.all(caer_gray_rgb == caer_hsv_rgb) 
    # assert np.all(caer_hsv_rgb == caer_hls_rgb) 
    # assert np.all(caer_hls_rgb == caer_lab_rgb)


def test_to_bgr():
    caer_bgr_bgr = caer.to_bgr(cv_bgr)
    caer_rgb_bgr = caer.to_bgr(cv_rgb)
    caer_gray_bgr = caer.to_bgr(cv_gray)
    caer_hsv_bgr = caer.to_bgr(cv_hsv)
    caer_hls_bgr = caer.to_bgr(cv_hls)
    caer_lab_bgr = caer.to_bgr(cv_lab)

    assert isinstance(caer_bgr_bgr, caer.Tensor)
    assert isinstance(caer_rgb_bgr, caer.Tensor)
    assert isinstance(caer_gray_bgr, caer.Tensor)
    assert isinstance(caer_hsv_bgr, caer.Tensor)
    assert isinstance(caer_hls_bgr, caer.Tensor)
    assert isinstance(caer_lab_bgr, caer.Tensor)

    assert caer_bgr_bgr.shape == cv_bgr.shape 
    assert caer_rgb_bgr.shape == cv_bgr.shape 
    assert caer_gray_bgr.shape == cv_bgr.shape 
    assert caer_hsv_bgr.shape == cv_bgr.shape 
    assert caer_hls_bgr.shape == cv_bgr.shape 
    assert caer_lab_bgr.shape == cv_bgr.shape 

    assert caer_bgr_bgr.cspace == 'bgr'
    assert caer_rgb_bgr.cspace == 'bgr'
    assert caer_gray_bgr.cspace == 'bgr'
    assert caer_hsv_bgr.cspace == 'bgr'
    assert caer_hls_bgr.cspace == 'bgr'
    assert caer_lab_bgr.cspace == 'bgr'

    assert len(caer_bgr_bgr.shape) == 3
    assert len(caer_rgb_bgr.shape) == 3
    assert len(caer_gray_bgr.shape) == 3
    assert len(caer_hsv_bgr.shape) == 3
    assert len(caer_hls_bgr.shape) == 3
    assert len(caer_lab_bgr.shape) == 3

    # assert np.all(caer_bgr_bgr == caer_rgb_bgr) 
    # assert np.all(caer_rgb_bgr == caer_hsv_bgr) 
    # assert np.all(caer_rgb_bgr == caer_gray_bgr) 
    # assert np.all(caer_gray_bgr == caer_hsv_bgr) 
    # assert np.all(caer_hsv_bgr == caer_hls_bgr) 
    # assert np.all(caer_hls_bgr == caer_lab_bgr)


def test_to_gray():
    caer_gray_gray = caer.to_gray(cv_gray)
    caer_bgr_gray = caer.to_gray(cv_bgr)
    caer_rgb_gray = caer.to_gray(cv_rgb)
    caer_hsv_gray = caer.to_gray(cv_hsv)
    caer_hls_gray = caer.to_gray(cv_hls)
    caer_lab_gray = caer.to_gray(cv_lab)

    assert isinstance(caer_gray_gray, caer.Tensor)
    assert isinstance(caer_bgr_gray, caer.Tensor)
    assert isinstance(caer_rgb_gray, caer.Tensor)
    assert isinstance(caer_hsv_gray, caer.Tensor)
    assert isinstance(caer_hls_gray, caer.Tensor)
    assert isinstance(caer_lab_gray, caer.Tensor)

    assert caer_gray_gray.shape == cv_gray.shape 
    assert caer_bgr_gray.shape == cv_gray.shape 
    assert caer_rgb_gray.shape == cv_gray.shape 
    assert caer_hsv_gray.shape == cv_gray.shape 
    assert caer_hls_gray.shape == cv_gray.shape 
    assert caer_lab_gray.shape == cv_gray.shape 

    assert caer_gray_gray.cspace == 'gray'
    assert caer_bgr_gray.cspace == 'gray'
    assert caer_rgb_gray.cspace == 'gray'
    assert caer_hsv_gray.cspace == 'gray'
    assert caer_hls_gray.cspace == 'gray'
    assert caer_lab_gray.cspace == 'gray'

    assert len(caer_gray_gray.shape) == 2
    assert len(caer_bgr_gray.shape) == 2
    assert len(caer_rgb_gray.shape) == 2
    assert len(caer_hsv_gray.shape) == 2
    assert len(caer_hls_gray.shape) == 2
    assert len(caer_lab_gray.shape) == 2

    # assert np.all(caer_gray_gray == caer_bgr_gray) 
    # assert np.all(caer_bgr_gray == caer_hsv_gray) 
    # assert np.all(caer_bgr_gray == caer_rgb_gray) 
    # assert np.all(caer_rgb_gray == caer_hsv_gray) 
    # assert np.all(caer_hsv_gray == caer_hls_gray) 
    # assert np.all(caer_hls_gray == caer_lab_gray)


def test_to_hsv():
    caer_hsv_hsv = caer.to_hsv(cv_hsv)
    caer_bgr_hsv = caer.to_hsv(cv_bgr)
    caer_gray_hsv = caer.to_hsv(cv_gray)
    caer_rgb_hsv = caer.to_hsv(cv_rgb)
    caer_hls_hsv = caer.to_hsv(cv_hls)
    caer_lab_hsv = caer.to_hsv(cv_lab)

    assert isinstance(caer_hsv_hsv, caer.Tensor)
    assert isinstance(caer_bgr_hsv, caer.Tensor)
    assert isinstance(caer_gray_hsv, caer.Tensor)
    assert isinstance(caer_rgb_hsv, caer.Tensor)
    assert isinstance(caer_hls_hsv, caer.Tensor)
    assert isinstance(caer_lab_hsv, caer.Tensor)

    assert caer_hsv_hsv.shape == cv_hsv.shape 
    assert caer_bgr_hsv.shape == cv_hsv.shape 
    assert caer_gray_hsv.shape == cv_hsv.shape 
    assert caer_rgb_hsv.shape == cv_hsv.shape 
    assert caer_hls_hsv.shape == cv_hsv.shape 
    assert caer_lab_hsv.shape == cv_hsv.shape 

    assert caer_hsv_hsv.cspace == 'hsv'
    assert caer_bgr_hsv.cspace == 'hsv'
    assert caer_gray_hsv.cspace == 'hsv'
    assert caer_rgb_hsv.cspace == 'hsv'
    assert caer_hls_hsv.cspace == 'hsv'
    assert caer_lab_hsv.cspace == 'hsv'

    assert len(caer_hsv_hsv.shape) == 3
    assert len(caer_bgr_hsv.shape) == 3
    assert len(caer_gray_hsv.shape) == 3
    assert len(caer_rgb_hsv.shape) == 3
    assert len(caer_hls_hsv.shape) == 3
    assert len(caer_lab_hsv.shape) == 3

    # assert np.all(caer_hsv_hsv == caer_bgr_hsv) 
    # assert np.all(caer_bgr_hsv == caer_rgb_hsv) 
    # assert np.all(caer_bgr_hsv == caer_gray_hsv) 
    # assert np.all(caer_gray_hsv == caer_rgb_hsv) 
    # assert np.all(caer_rgb_hsv == caer_hls_hsv) 
    # assert np.all(caer_hls_hsv == caer_lab_hsv)


def test_to_hls():
    caer_hls_hls = caer.to_hls(cv_hls)
    caer_bgr_hls = caer.to_hls(cv_bgr)
    caer_gray_hls = caer.to_hls(cv_gray)
    caer_hsv_hls = caer.to_hls(cv_hsv)
    caer_rgb_hls = caer.to_hls(cv_rgb)
    caer_lab_hls = caer.to_hls(cv_lab)

    assert isinstance(caer_hls_hls, caer.Tensor)
    assert isinstance(caer_bgr_hls, caer.Tensor)
    assert isinstance(caer_gray_hls, caer.Tensor)
    assert isinstance(caer_hsv_hls, caer.Tensor)
    assert isinstance(caer_rgb_hls, caer.Tensor)
    assert isinstance(caer_lab_hls, caer.Tensor)

    assert caer_hls_hls.shape == cv_hls.shape 
    assert caer_bgr_hls.shape == cv_hls.shape 
    assert caer_gray_hls.shape == cv_hls.shape 
    assert caer_hsv_hls.shape == cv_hls.shape 
    assert caer_rgb_hls.shape == cv_hls.shape 
    assert caer_lab_hls.shape == cv_hls.shape 

    assert caer_hls_hls.cspace == 'hls'
    assert caer_bgr_hls.cspace == 'hls'
    assert caer_gray_hls.cspace == 'hls'
    assert caer_hsv_hls.cspace == 'hls'
    assert caer_rgb_hls.cspace == 'hls'
    assert caer_lab_hls.cspace == 'hls'

    assert len(caer_hls_hls.shape) == 3
    assert len(caer_bgr_hls.shape) == 3
    assert len(caer_gray_hls.shape) == 3
    assert len(caer_hsv_hls.shape) == 3
    assert len(caer_rgb_hls.shape) == 3
    assert len(caer_lab_hls.shape) == 3

    # assert np.all(caer_hls_hls == caer_bgr_hls) 
    # assert np.all(caer_bgr_hls == caer_hsv_hls) 
    # assert np.all(caer_bgr_hls == caer_gray_hls) 
    # assert np.all(caer_gray_hls == caer_hsv_hls) 
    # assert np.all(caer_hsv_hls == caer_rgb_hls) 
    # assert np.all(caer_rgb_hls == caer_lab_hls)


def test_to_lab():
    caer_lab_lab = caer.to_lab(cv_lab)
    caer_bgr_lab = caer.to_lab(cv_bgr)
    caer_gray_lab = caer.to_lab(cv_gray)
    caer_hsv_lab = caer.to_lab(cv_hsv)
    caer_hls_lab = caer.to_lab(cv_hls)
    caer_rgb_lab = caer.to_lab(cv_rgb)

    assert isinstance(caer_lab_lab, caer.Tensor)
    assert isinstance(caer_bgr_lab, caer.Tensor)
    assert isinstance(caer_gray_lab, caer.Tensor)
    assert isinstance(caer_hsv_lab, caer.Tensor)
    assert isinstance(caer_hls_lab, caer.Tensor)
    assert isinstance(caer_rgb_lab, caer.Tensor)

    assert caer_lab_lab.shape == cv_lab.shape 
    assert caer_bgr_lab.shape == cv_lab.shape 
    assert caer_gray_lab.shape == cv_lab.shape 
    assert caer_hsv_lab.shape == cv_lab.shape 
    assert caer_hls_lab.shape == cv_lab.shape 
    assert caer_rgb_lab.shape == cv_lab.shape 

    assert caer_lab_lab.cspace == 'lab'
    assert caer_bgr_lab.cspace == 'lab'
    assert caer_gray_lab.cspace == 'lab'
    assert caer_hsv_lab.cspace == 'lab'
    assert caer_hls_lab.cspace == 'lab'
    assert caer_rgb_lab.cspace == 'lab'

    assert len(caer_lab_lab.shape) == 3
    assert len(caer_bgr_lab.shape) == 3
    assert len(caer_gray_lab.shape) == 3
    assert len(caer_hsv_lab.shape) == 3
    assert len(caer_hls_lab.shape) == 3
    assert len(caer_rgb_lab.shape) == 3

    # assert np.all(caer_lab_lab == caer_bgr_lab) 
    # assert np.all(caer_bgr_lab == caer_hsv_lab) 
    # assert np.all(caer_bgr_lab == caer_gray_lab) 
    # assert np.all(caer_gray_lab == caer_hsv_lab) 
    # assert np.all(caer_hsv_lab == caer_hls_lab) 
    # assert np.all(caer_hls_lab == caer_rgb_lab)