#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv 
import numpy as np 

from .constants import GRAY2BGR, GRAY2RGB
from .bgr import bgr_to_lab, bgr_to_hsv

__all__ = [
    'gray_to_rgb',
    'gray_to_bgr',
    'gray_to_hsv',
    'gray_to_lab',
]


def gray_to_rgb(img) -> np.ndarray:
    """
        Converts an Grayscale image to its RGB version
    """
    if len(img.shape) != 1:
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its RGB counterpart')

    return cv.cvtColor(img, GRAY2RGB)


def gray_to_bgr(img) -> np.ndarray:
    """
        Converts am Grayscale image to its BGR version
    """
    if len(img.shape) != 1:
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its BGR counterpart')

    return cv.cvtColor(img, GRAY2BGR)


def gray_to_lab(img) -> np.ndarray:
    """
        Converts an Grayscale image to its LAB version
    """
    if len(img.shape) != 3:
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a Grayscale image to its LAB counterpart')

    bgr = gray_to_bgr(img)

    return bgr_to_lab(bgr)


def gray_to_hsv(img) -> np.ndarray:
    """
        Converts an Grayscale image to its HSV version
    """
    if len(img.shape) != 1:
        raise ValueError(f'Image of shape 3 expected. Found shape {len(img.shape)}. This method converts a LAB image to its HSV counterpart')

    bgr = gray_to_bgr(img)

    return bgr_to_hsv(bgr)

