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
import os 
import cv2 as cv 
import numpy as np 

here = os.path.dirname(os.path.dirname(__file__))
tens_path = os.path.join(here, 'data', 'green_fish.jpg')


def test_imread():
    cv_bgr = cv.imread(tens_path) 
    cv_rgb = cv.cvtColor(cv_bgr.copy(), cv.COLOR_BGR2RGB)
    

    caer_bgr = caer.imread(tens_path, rgb=False) # Expected: BGR
    caer_rgb = caer.imread(tens_path, rgb=True) # Expected: RGB

    # caer_rgbF_grayF = caer.imread(tens_path, rgb=False, gray=False) # Expected: BGR
    # caer_rgbT_grayF = caer.imread(tens_path, rgb=True, gray=False) # Expected: RGB
    # caer_rgbT_grayT = caer.imread(tens_path, rgb=True, gray=True) # Expected: Gray
    # caer_rgbF_grayT = caer.imread(tens_path, rgb=False, gray=True) # Expected: Gray
    # caer_rgbT_grayT = caer.imread(tens_path, rgb=True, gray=True) # Expected: Gray


    # Asserts
    assert np.all(caer_bgr == cv_bgr)
    assert np.all(caer_rgb == cv_rgb)

    # assert np.all(caer_rgbF_grayF == cv_bgr)
    # assert np.all(caer_rgbT_grayF == cv_rgb)
    # assert np.all(caer_rgbT_grayT == cv_gray)
    # assert np.all(caer_rgbF_grayT == cv_gray)
    # assert np.all(caer_rgbT_grayT == cv_gray)


def test_imread_target_sizes():
    tens_400_400 = caer.imread(tens_path, target_size=(400,400))
    tens_304_339 = caer.imread(tens_path, target_size=(304,339))
    tens_199_206 = caer.imread(tens_path, target_size=(199,206))

    assert tens_400_400.shape[:2] == (400,400)
    assert tens_304_339.shape[:2] == (339,304) # Numpy arrays are processed differently (h,w) as opposed to (w,h)
    assert tens_199_206.shape[:2] == (206,199) # Numpy arrays are processed differently (h,w) as opposed to (w,h)

    assert isinstance(tens_400_400, caer.Tensor)
    assert isinstance(tens_304_339, caer.Tensor)
    assert isinstance(tens_199_206, caer.Tensor)


def test_imread_preserve_aspect_ratio():
    tens_400_400 = caer.imread(tens_path, target_size=(400,400), preserve_aspect_ratio=True)
    tens_223_182 = caer.imread(tens_path, target_size=(223,182), preserve_aspect_ratio=True)
    tens_93_35 = caer.imread(tens_path, target_size=(93,35), preserve_aspect_ratio=True)

    assert tens_400_400.shape[:2] == (400,400)
    assert tens_223_182.shape[:2] == (182,223) # Numpy arrays are processed differently (h,w) as opposed to (w,h)
    assert tens_93_35.shape[:2] == (35,93) # Numpy arrays are processed differently (h,w) as opposed to (w,h)

    assert isinstance(tens_400_400, caer.Tensor)
    assert isinstance(tens_223_182, caer.Tensor)
    assert isinstance(tens_93_35, caer.Tensor)