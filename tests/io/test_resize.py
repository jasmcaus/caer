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
cv_img = cv.imread(img_path)
caer_img = caer.imread(img_path)


def test_target_sizes():
    # Should return <ndarray>s
    np_img_400_400 = caer.resize(cv_img, target_size=(400,400))
    np_img_304_339 = caer.resize(cv_img, target_size=(304,339))
    np_img_199_206 = caer.resize(cv_img, target_size=(199,206))

    # Should return <caer.Tensor>s
    caer_img_400_400 = caer.resize(caer_img, target_size=(400,400))
    caer_img_304_339 = caer.resize(caer_img, target_size=(304,339))
    caer_img_199_206 = caer.resize(caer_img, target_size=(199,206))

    assert np_img_400_400.shape[:2] == (400,400)
    assert np_img_304_339.shape[:2] == (339,304)
    assert np_img_199_206.shape[:2] == (206,199)

    assert caer_img_400_400.shape[:2] == (400,400)
    assert caer_img_304_339.shape[:2] == (339,304)
    assert caer_img_199_206.shape[:2] == (206,199)


    # Type Asserts
    ## Using isinstance() often mistakes a caer.Tensor as an np.ndarray
    assert 'numpy.ndarray' in str(type(np_img_400_400))
    assert 'numpy.ndarray' in str(type(np_img_304_339))
    assert 'numpy.ndarray' in str(type(np_img_199_206))

    assert 'caer.Tensor' in str(type(caer_img_400_400))
    assert 'caer.Tensor' in str(type(caer_img_304_339))
    assert 'caer.Tensor' in str(type(caer_img_199_206))


def test_preserve_aspect_ratio():

    np_img_400_400 = caer.resize(cv_img, target_size=(400,400), preserve_aspect_ratio=True)
    np_img_223_182 = caer.resize(cv_img, target_size=(223,182), preserve_aspect_ratio=True)
    np_img_93_35 = caer.resize(cv_img, target_size=(93,35), preserve_aspect_ratio=True)

    caer_img_400_400 = caer.resize(caer_img, target_size=(400,400), preserve_aspect_ratio=True)
    caer_img_223_182 = caer.resize(caer_img, target_size=(223,182), preserve_aspect_ratio=True)
    caer_img_93_35 = caer.resize(caer_img, target_size=(93,35), preserve_aspect_ratio=True)

    assert np_img_400_400.shape[:2] == (400,400)
    assert np_img_223_182.shape[:2] == (182,223)
    assert np_img_93_35.shape[:2] == (35,93)

    assert caer_img_400_400.shape[:2] == (400,400)
    assert caer_img_223_182.shape[:2] == (182,223)
    assert caer_img_93_35.shape[:2] == (35,93)


    # Type Asserts
    ## Using isinstance() often mistakes a caer.Tensor as an np.ndarray
    assert 'numpy.ndarray' in str(type(np_img_400_400))
    assert 'numpy.ndarray' in str(type(np_img_223_182))
    assert 'numpy.ndarray' in str(type(np_img_93_35))

    assert 'caer.Tensor' in str(type(caer_img_400_400))
    assert 'caer.Tensor' in str(type(caer_img_223_182))
    assert 'caer.Tensor' in str(type(caer_img_93_35))