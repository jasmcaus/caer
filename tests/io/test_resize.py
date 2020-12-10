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

here = os.path.dirname(os.path.dirname(__file__))

def test_target_sizes():
    test_img = os.path.join(here, 'data', 'beverages.jpg')

    img_400_400 = caer.imread(test_img, target_size=(400,400))
    img_304_339 = caer.imread(test_img, target_size=(304,339))
    img_199_206 = caer.imread(test_img, target_size=(199,206))

    assert img_400_400.shape[:2] == (400,400)
    assert img_304_339.shape[:2] == (339,304) # Numpy arrays are processes differently (h,w) as opposed to (w,h)
    assert img_199_206.shape[:2] == (206,199) # Numpy arrays are processes differently (h,w) as opposed to (w,h)


def test_preserve_aspect_ratio():
    test_img = os.path.join(here, 'data', 'green_fish.jpg')

    img_400_400 = caer.imread(test_img, target_size=(400,400), preserve_aspect_ratio=True)
    img_223_182 = caer.imread(test_img, target_size=(223,182), preserve_aspect_ratio=True)
    img_93_35 = caer.imread(test_img, target_size=(93,35), preserve_aspect_ratio=True)

    assert img_400_400.shape[:2] == (400,400)
    assert img_223_182.shape[:2] == (182,223) # Numpy arrays are processes differently (h,w) as opposed to (w,h)
    assert img_93_35.shape[:2] == (35,93) # Numpy arrays are processes differently (h,w) as opposed to (w,h)