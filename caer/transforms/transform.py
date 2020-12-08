#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
import cv2 as cv 

__all__ = [
    'hflip',
    'vflip',
    'hvflip'
]


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def vflip(img):
    return np.ascontiguousarray(img[::-1, ...])


def hvflip(img):
    return hflip(vflip(img))