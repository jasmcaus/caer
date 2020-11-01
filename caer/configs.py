# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

import cv2 as cv 

# Configuration Variables used in Caer
CROP_CENTRE = 1
CROP_TOP = 2
CROP_LEFT = 3
CROP_RIGHT = 4
CROP_BOTTOM = 5

VALID_URL_NO_EXIST = 0
INVALID_URL_STRING = -1

# OpenCV
BGR2GRAY = cv.COLOR_BGR2GRAY
BGR2RGB = cv.COLOR_BGR2RGB
BGR2HSV = cv.COLOR_BGR2HSV
BGR2LAB = cv.COLOR_BGR2LAB
RGB2GRAY = cv.COLOR_RGB2GRAY
RGB2HSV = cv.COLOR_RGB2HSV
RGB2LAB = cv.COLOR_RGB2LAB
IMREAD_COLOR = cv.IMREAD_COLOR

INTER_NEAREST = cv.INTER_NEAREST
INTER_LINEAR = cv.INTER_LINEAR
INTER_CUBIC = cv.INTER_CUBIC
INTER_AREA = cv.INTER_AREA

#pylint:disable=c-extension-no-member
FRAME_COUNT = cv.CAP_PROP_FRAME_COUNT
FRAME_COUNT_DEPR = cv.cv2.CAP_PROP_FRAME_COUNT
FPS = cv.CAP_PROP_FPS
FPS_DEPR = cv.cv2.CAP_PROP_FPS

__all__ = [
    'CROP_CENTRE',
    'CROP_TOP',
    'CROP_LEFT',
    'CROP_RIGHT',
    'CROP_BOTTOM',
    'VALID_URL_NO_EXIST',
    'INVALID_URL_STRING',
]