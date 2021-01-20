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
import os 
import cv2 as cv 
import numpy as np 

here = os.path.dirname(os.path.dirname(__file__))


def test_imread():
    img_path = os.path.join(here, 'data', 'beverages.jpg')

    cv_bgr = cv.imread(img_path) 
    cv_rgb = cv.cvtColor(cv_bgr.copy(), cv.COLOR_BGR2RGB)
    

    caer_bgr = caer.imread(img_path, rgb=False) # Expected: BGR
    caer_rgb = caer.imread(img_path, rgb=True) # Expected: RGB

    # caer_rgbF_grayF = caer.imread(img_path, rgb=False, gray=False) # Expected: BGR
    # caer_rgbT_grayF = caer.imread(img_path, rgb=True, gray=False) # Expected: RGB
    # caer_rgbT_grayT = caer.imread(img_path, rgb=True, gray=True) # Expected: Gray
    # caer_rgbF_grayT = caer.imread(img_path, rgb=False, gray=True) # Expected: Gray
    # caer_rgbT_grayT = caer.imread(img_path, rgb=True, gray=True) # Expected: Gray


    # Asserts
    assert np.all(caer_bgr == cv_bgr)
    assert np.all(caer_rgb == cv_rgb)

    # assert np.all(caer_rgbF_grayF == cv_bgr)
    # assert np.all(caer_rgbT_grayF == cv_rgb)
    # assert np.all(caer_rgbT_grayT == cv_gray)
    # assert np.all(caer_rgbF_grayT == cv_gray)
    # assert np.all(caer_rgbT_grayT == cv_gray)


def test_imread_target_sizes():
    img_path = os.path.join(here, 'data', 'beverages.jpg')

    img_400_400 = caer.imread(img_path, target_size=(400,400))
    img_304_339 = caer.imread(img_path, target_size=(304,339))
    img_199_206 = caer.imread(img_path, target_size=(199,206))

    assert img_400_400.shape[:2] == (400,400)
    assert img_304_339.shape[:2] == (339,304) # Numpy arrays are processed differently (h,w) as opposed to (w,h)
    assert img_199_206.shape[:2] == (206,199) # Numpy arrays are processed differently (h,w) as opposed to (w,h)


def test_imread_preserve_aspect_ratio():
    img_path = os.path.join(here, 'data', 'green_fish.jpg')

    img_400_400 = caer.imread(img_path, target_size=(400,400), preserve_aspect_ratio=True)
    img_223_182 = caer.imread(img_path, target_size=(223,182), preserve_aspect_ratio=True)
    img_93_35 = caer.imread(img_path, target_size=(93,35), preserve_aspect_ratio=True)

    assert img_400_400.shape[:2] == (400,400)
    assert img_223_182.shape[:2] == (182,223) # Numpy arrays are processed differently (h,w) as opposed to (w,h)
    assert img_93_35.shape[:2] == (35,93) # Numpy arrays are processed differently (h,w) as opposed to (w,h)


# def test_gray():
#     img_path = os.path.join(here, 'data', 'green_fish.jpg')

#     img = caer.imread(img_path, gray=True)
#     img_chann = caer.imread(img_path, gray=True) # Maintain backwards-compatibility

#     assert len(img.shape) == 2
#     assert len(img_chann.shape) == 2