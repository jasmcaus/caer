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
import os 
import cv2 as cv 
import numpy as np 

here = os.path.dirname(os.path.dirname(__file__))


def test_imread():
    test_img = os.path.join(here, 'data', 'beverages.jpg')

    img = caer.imread(test_img)
    test_against = cv.imread(test_img) 

    assert np.all(img == test_against)


def test_gray():
    test_img = os.path.join(here, 'data', 'green_fish.jpg')

    img = caer.imread(test_img, channels=1)

    assert len(img.shape) == 2