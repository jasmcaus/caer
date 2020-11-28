#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


__all__ = [
    'CROP_CENTRE',
    'CROP_TOP',
    'CROP_LEFT',
    'CROP_RIGHT',
    'CROP_BOTTOM',
    'VALID_URL_NO_EXIST',
    'INVALID_URL_STRING',
    'BGR2GRAY',
    'BGR2RGB',
    'BGR2HSV',
    'BGR2LAB',
    'RGB2GRAY',
    'RGB2HSV',
    'RGB2LAB',
    'IMREAD_COLOR',
    'INTER_NEAREST',
    'INTER_LINEAR',
    'INTER_AREA',
    'INTER_CUBIC',
    'FRAME_COUNT',
    'FPS'
]


# Configuration Variables used in Caer
CROP_CENTRE = 1
CROP_TOP = 2
CROP_LEFT = 3
CROP_RIGHT = 4
CROP_BOTTOM = 5

VALID_URL_NO_EXIST = 0
INVALID_URL_STRING = -1

#  OpenCV 

## Interpolations
INTER_NEAREST = 0  # cv.INTER_NEAREST
INTER_LINEAR = 1  # cv.INTER_LINEAR
INTER_CUBIC = 2  # cv.INTER_CUBIC
INTER_AREA = 3  # cv.INTER_AREA

## Video
FPS = 5  # cv.CAP_PROP_FPS
FRAME_COUNT = 7  # cv.CAP_PROP_FRAME_COUNT
