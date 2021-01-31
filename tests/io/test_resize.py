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
cv_tens = cv.imread(tens_path)
caer_tens = caer.imread(tens_path)


def test_target_sizes():
    # Should return <ndarray>s
    tens_400_400 = caer.resize(cv_tens, target_size=(400,400))
    tens_304_339 = caer.resize(cv_tens, target_size=(304,339))
    tens_199_206 = caer.resize(cv_tens, target_size=(199,206))

    # Should return <caer.Tensor>s
    caer_tens_400_400 = caer.resize(caer_tens, target_size=(400,400))
    caer_tens_304_339 = caer.resize(caer_tens, target_size=(304,339))
    caer_tens_199_206 = caer.resize(caer_tens, target_size=(199,206))

    assert tens_400_400.shape[:2] == (400,400)
    assert tens_304_339.shape[:2] == (339,304)
    assert tens_199_206.shape[:2] == (206,199)


    # Type Asserts
    ## Using isinstance() often mistakes a caer.Tensor as an np.ndarray
    assert 'caer.Tensor' in str(type(tens_400_400))
    assert 'caer.Tensor' in str(type(tens_304_339))
    assert 'caer.Tensor' in str(type(tens_199_206))



def test_preserve_aspect_ratio():

    tens_400_400 = caer.resize(cv_tens, target_size=(400,400), preserve_aspect_ratio=True)
    tens_223_182 = caer.resize(cv_tens, target_size=(223,182), preserve_aspect_ratio=True)
    tens_93_35 = caer.resize(cv_tens, target_size=(93,35), preserve_aspect_ratio=True)

    assert tens_400_400.shape[:2] == (400,400)
    assert tens_223_182.shape[:2] == (182,223)
    assert tens_93_35.shape[:2] == (35,93)


    # Type Asserts
    ## Using isinstance() often mistakes a caer.Tensor as an np.ndarray
    assert 'caer.Tensor' in str(type(tens_400_400))
    assert 'caer.Tensor' in str(type(tens_223_182))
    assert 'caer.Tensor' in str(type(tens_93_35))